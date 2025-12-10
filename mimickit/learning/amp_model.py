import torch

import learning.nets.net_builder as net_builder
import learning.ppo_model as ppo_model
import util.torch_util as torch_util

class AMPModel(ppo_model.PPOModel):
    def __init__(self, config, env):
        super().__init__(config, env)
        return
    
    def eval_disc(self, disc_obs):
        h = self._disc_layers(disc_obs)
        val = self._disc_logits(h)
        return val

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)
    
    def get_disc_weights(self):
        weights = []
        for m in self._disc_layers.modules():
            if hasattr(m, "weight"):
                weights.append(torch.flatten(m.weight))

        weights.append(torch.flatten(self._disc_logits.weight))
        return weights

    def _build_nets(self, config, env):
        super()._build_nets(config, env)
        self._build_disc(config, env)
        return

    def _build_disc(self, config, env):
        init_output_scale = 1.0
        net_name = config["disc_net"]

        input_dict = self._build_disc_input_dict(env)
        self._disc_layers, layers_info = net_builder.build_net(net_name, input_dict,
                                                                 activation=self._activation)

        layers_out_size = torch_util.calc_layers_out_size(self._disc_layers)
        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -init_output_scale, init_output_scale)
        torch.nn.init.zeros_(self._disc_logits.bias)
        return

    def _build_disc_input_dict(self, env):
        obs_space = env.get_disc_obs_space()
        input_dict = {"disc_obs": obs_space}
        return input_dict
    