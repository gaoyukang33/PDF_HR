import engines.engine_builder as engine_builder

import abc
import envs.base_env as base_env
import gymnasium.spaces as spaces
import numpy as np
import torch

import util.torch_util as torch_util

class SimEnv(base_env.BaseEnv):
    NAME = "sim_env"

    def __init__(self, config, num_envs, device, visualize):
        super().__init__(visualize=visualize)

        self._device = device

        env_config = config["env"]
        self._episode_length = env_config["episode_length"] # episode length in seconds
        
        engine_config = config["engine"]
        self._engine = self._build_engine(engine_config, num_envs, device, visualize)
        self._build_envs(config, num_envs)
        self._engine.initialize_sim()
        
        self._action_space = self._build_action_space()
        self._build_sim_tensors(config)
        self._build_data_buffers()

        if self._visualize:
            self._init_camera()

        return
    
    def get_obs_space(self):
        obs = self._compute_obs()
        obs_shape = list(obs.shape[1:])
        obs_dtype = torch_util.torch_dtype_to_numpy(obs.dtype)
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=obs_dtype,
        )
        return obs_space
    
    def reset(self, env_ids=None):
        if (env_ids is None):
            num_envs = self.get_num_envs()
            reset_env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)
        else:
            reset_env_ids = env_ids

        self._reset_envs(reset_env_ids)
        self._engine.update_sim_state()

        self._update_observations(env_ids)
        self._update_info(env_ids)

        return self._obs_buf, self._info
    
    def step(self, action):
        self._pre_physics_step(action)

        self._physics_step()
        
        # compute observations, rewards, resets, ...
        self._post_physics_step()

        if (self._visualize):
            self._render()
        
        return self._obs_buf, self._reward_buf, self._done_buf, self._info
    
    def get_num_envs(self):
        return self._engine.get_num_envs()
    
    def get_env_time(self, env_ids=None):
        if (env_ids is None):
            env_time = self._time_buf
        else:
            env_time = self._time_buf[env_ids]
        return env_time
    
    def _pre_physics_step(self, actions):
        self._apply_action(actions)
        return
    
    def _physics_step(self):
        self._step_sim()
        return
    
    def _render(self):
        self._update_camera()
        self._engine.render()
        return
    
    def _step_sim(self):
        self._engine.step()
        return
    
    def _update_misc(self):
        return

    def _update_info(self, env_ids=None):
        return
    
    def _update_camera(self):
        return

    @abc.abstractmethod
    def _apply_action(self, actions):
        return
    
    @abc.abstractmethod
    def _update_reward(self):
        return
    
    @abc.abstractmethod
    def _update_done(self):
        return
    
    @abc.abstractmethod
    def _compute_obs(env_ids=None):
        return
    
    def _update_observations(self, env_ids=None):
        if (env_ids is None or len(env_ids) > 0):
            obs = self._compute_obs(env_ids)
            if (env_ids is None):
                self._obs_buf[:] = obs
            else:
                self._obs_buf[env_ids] = obs
        return

    def _post_physics_step(self):
        self._update_time()
        self._update_misc()
        self._update_observations()
        self._update_info()
        self._update_reward()
        self._update_done()
        return

    def _build_engine(self, engine_config, num_envs, device, visualize):
        engine = engine_builder.build_engine(engine_config, num_envs, device, visualize)
        return engine
    
    @abc.abstractmethod
    def _build_envs(self, config, num_envs):
        return
    
    def _build_sim_tensors(self, config):
        return

    def _build_data_buffers(self):
        num_envs = self.get_num_envs()

        self._reward_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)
        self._done_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._timestep_buf = torch.zeros(num_envs, device=self._device, dtype=torch.int)
        self._time_buf = torch.zeros(num_envs, device=self._device, dtype=torch.float)

        obs_space = self.get_obs_space()
        obs_dtype = torch_util.numpy_dtype_to_torch(obs_space.dtype)
        self._obs_buf = torch.zeros([num_envs] + list(obs_space.shape), device=self._device, dtype=obs_dtype)

        self._info = dict()
        return
    
    @abc.abstractmethod
    def _reset_envs(self, env_ids):
        if (len(env_ids) > 0):
            self._timestep_buf[env_ids] = 0
            self._time_buf[env_ids] = 0
            self._done_buf[env_ids] = base_env.DoneFlags.NULL.value
        return
    
    def _update_time(self):
        self._timestep_buf += 1
        self._time_buf[:] = self._engine.get_timestep() * self._timestep_buf
        return

    @abc.abstractmethod
    def _build_action_space(self):
        return
    
    @abc.abstractmethod
    def _init_camera(self):
        return
