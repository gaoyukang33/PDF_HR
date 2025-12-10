import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.nets.net_builder as net_builder
import learning.amp_model as amp_model
import util.torch_util as torch_util

class ASEModel(amp_model.AMPModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        return
    
    def eval_actor(self, obs, z):
        in_data = torch.cat([obs, z], dim=-1)
        h = self._actor_layers(in_data)
        a_dist = self._action_dist(h)
        return a_dist
    
    def eval_critic(self, obs, z):
        in_data = torch.cat([obs, z], dim=-1)
        h = self._critic_layers(in_data)
        val = self._critic_out(h)
        return val
    
    def eval_enc(self, enc_obs):
        h = self._enc_layers(enc_obs)
        unorm_z = self._enc_out(h)
        z = torch.nn.functional.normalize(unorm_z, dim=-1)
        return z
    
    def get_enc_weights(self):
        weights = []
        for m in self._enc_layers.modules():
            if hasattr(m, "weight"):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._enc_out.weight))
        return weights

    def get_latent_dim(self):
        return self._enc_out.out_features

    def _build_nets(self, config, env):
        self._build_enc(config, env)
        super()._build_nets(config, env)
        return
    
    def _build_actor_input_dict(self, env):
        obs_space = env.get_obs_space()
        z_space = self._build_latent_space()
        input_dict = {"obs": obs_space,
                      "z": z_space}
        return input_dict
    
    def _build_critic_input_dict(self, env):
        obs_space = env.get_obs_space()
        z_space = self._build_latent_space()
        input_dict = {"obs": obs_space,
                      "z": z_space}
        return input_dict

    def _build_enc(self, config, env):
        enc_logit_init_scale = 0.1

        net_name = config["enc_net"]
        latent_dim = config["latent_dim"]

        input_dict = self._build_enc_input_dict(env)
        self._enc_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                 activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._enc_layers)
        self._enc_out = torch.nn.Linear(layers_out_size, latent_dim)

        torch.nn.init.uniform_(self._enc_out.weight, -enc_logit_init_scale, enc_logit_init_scale)
        torch.nn.init.zeros_(self._enc_out.bias)
        return

    def _build_enc_input_dict(self, env):
        obs_space = env.get_disc_obs_space()
        input_dict = {"enc_obs": obs_space}
        return input_dict
    
    def _build_latent_space(self):
        z_dim = self.get_latent_dim()
        z_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=[z_dim],
            dtype=np.float32,
        )
        return z_space
