import gymnasium.spaces as spaces
import torch

import learning.base_agent as base_agent
import util.torch_util as torch_util


class DummyAgent(base_agent.BaseAgent):
    NAME = "dummy"

    def __init__(self, env, device):
        super().__init__(None, env, device)
        return

    def _get_exp_buffer_length(self):
        return 32
    
    def _load_params(self, config):
        self._discount = 0.99
        self._iters_per_output = 100
        self._normalizer_samples = 10000
        self._test_episodes = 10
        self._steps_per_iter = 32
        return

    def _build_optimizer(self, config):
        return

    def _decide_action(self, obs, info):
        num_envs = obs.shape[0]
        a_space = self._env.get_action_space()
        a_dtype = torch_util.numpy_dtype_to_torch(a_space.dtype)

        if (isinstance(a_space, spaces.Box)):
            a_dim = a_space.low.shape[0]
            a = torch.zeros([num_envs, a_dim], device=self._device, dtype=a_dtype)
        elif (isinstance(a_space, spaces.Discrete)):
            a = torch.zeros([num_envs, 0], device=self._device, dtype=a_dtype)
        else:
            assert(False), "Unsuppoted action space: {}".format(a_space)

        a_info = dict()
        return a, a_info
    
    def _build_model(self, config):
        return
    
    def _update_model(self):
        info = dict()
        return info