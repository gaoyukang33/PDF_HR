import numpy as np
import torch

import util.mp_util as mp_util
from util.logger import Logger

class DiffNormalizer(torch.nn.Module):
    def __init__(self, shape, device, init_mean=None, min_diff=1e-4, clip=np.inf, dtype=torch.float):
        super().__init__()

        self._min_diff = min_diff
        self._clip = clip
        self.dtype = dtype
        self._build_params(shape, device, init_mean)
        return

    def record(self, x):
        shape = self.get_shape()
        assert len(x.shape) > len(shape)

        x = x.flatten(start_dim=0, end_dim=len(x.shape) - len(shape) - 1)

        self._new_count += x.shape[0]
        self._new_sum_abs += torch.sum(torch.abs(x), axis=0)
        return

    def update(self):
        self._new_count = mp_util.reduce_sum(self._new_count)
        mp_util.reduce_inplace_sum(self._new_sum_abs)

        new_count = self._new_count
        new_mean_abs = self._new_sum_abs / new_count

        new_total = self._count + new_count
        w_old = self._count.type(torch.float) / new_total.type(torch.float)
        w_new = float(new_count) / new_total.type(torch.float)

        self._mean_abs[:] = w_old * self._mean_abs + w_new * new_mean_abs
        self._count[:] = new_total

        self._new_count = 0
        self._new_sum_abs[:] = 0
        return

    def get_shape(self):
        return self._mean_abs.shape

    def get_count(self):
        return self._count

    def get_abs_mean(self):
        return self._mean_abs

    def normalize(self, x):
        diff = torch.clamp_min(self._mean_abs, self._min_diff)
        norm_x = x / diff
        norm_x = torch.clamp(norm_x, -self._clip, self._clip)
        return norm_x.type(self.dtype)

    def unnormalize(self, norm_x):
        diff = torch.clamp_min(self._mean_abs, self._min_diff)
        x = norm_x * diff
        return x.type(self.dtype)

    def _build_params(self, shape, device, init_mean):
        self._count = torch.nn.Parameter(torch.zeros([1], device=device, requires_grad=False, dtype=torch.long), requires_grad=False)
        self._mean_abs = torch.nn.Parameter(torch.ones(shape, device=device, requires_grad=False, dtype=self.dtype), requires_grad=False)

        if init_mean is not None:
            assert init_mean.shape == shape, \
            Logger.print('Normalizer init mean shape mismatch, expecting {:d}, but got {:d}'.shape(shape, init_mean.shape))
            self._mean_abs[:] = init_mean

        self._new_count = 0
        self._new_sum_abs = torch.zeros_like(self._mean_abs)
        return