import torch
import numpy as np

class CircularBuffer():
    def __init__(self, batch_size, buffer_len, shape, dtype, device):
        self._buffer = torch.zeros([batch_size, buffer_len] + list(shape), dtype=dtype, device=device)
        self._head = 0
        return

    def get_batch_size(self):
        return self._buffer.shape[0]

    def get_buffer_len(self):
        return self._buffer.shape[1]

    def push(self, data):
        self._buffer[:, self._head, ...] = data
        n = self.get_buffer_len()
        self._head = (self._head + 1) % n
        return

    def fill(self, batch_idx, data):
        buffer_len = self.get_buffer_len()
        self._buffer[batch_idx, self._head:, ...] = data[:, :buffer_len - self._head, ...]
        self._buffer[batch_idx, :self._head, ...] = data[:, buffer_len - self._head:, ...]
        return

    def get(self, idx):
        n = self.get_buffer_len()
        buffer_idx = self._head + idx

        if (torch.is_tensor(idx)):
            batch_size = self.get_batch_size()
            batch_idx = torch.arange(0, batch_size)
            buffer_idx = torch.remainder(buffer_idx, n)
            data = self._buffer[batch_idx, buffer_idx]
        else:
            buffer_idx = buffer_idx % n
            data = self._buffer[:, buffer_idx, ...]

        return data

    def get_all(self):
        if (self._head == 0):
            data = self._buffer
        else:
            n = self.get_buffer_len()
            idx = np.arange(self._head, self._head + n)
            idx = np.remainder(idx, n)
            data = self._buffer[:, idx, ...]

        return data

    def reset(self):
        self._head = 0
        return