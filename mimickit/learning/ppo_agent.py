import numpy as np
import torch

import envs.base_env as base_env
import learning.base_agent as base_agent
import learning.ppo_model as ppo_model
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util

class PPOAgent(base_agent.BaseAgent):
    NAME = "PPO"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        return

    def _load_params(self, config):
        super()._load_params(config)
        
        self._update_epochs = config["update_epochs"]
        self._batch_size = config["batch_size"]

        self._td_lambda = config["td_lambda"]
        self._ppo_clip_ratio = config["ppo_clip_ratio"]
        self._norm_adv_clip = config["norm_adv_clip"]

        self._action_bound_weight = config["action_bound_weight"]
        self._action_entropy_weight = config["action_entropy_weight"]
        self._action_reg_weight = config["action_reg_weight"]

        self._critic_loss_weight = config["critic_loss_weight"]
        self._critic_eval_batch_size = int(config.get("critic_eval_batch_size", 0))
        
        self._exp_anneal_samples = config.get("exp_anneal_samples", np.inf)
        self._exp_prob_beg = config.get("exp_prob_beg", 1.0)
        self._exp_prob_end = config.get("exp_prob_end", 1.0)
        
        return

    def _build_model(self, config):
        model_config = config["model"]
        self._model = ppo_model.PPOModel(model_config, self._env)
        return
    
    def _get_exp_buffer_length(self):
        return self._steps_per_iter
    
    def _init_iter(self):
        super()._init_iter()
        self._exp_buffer.reset()
        return

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        norm_action_dist = self._model.eval_actor(norm_obs)

        if (self._mode == base_agent.AgentMode.TRAIN):
            norm_a_rand = norm_action_dist.sample()
            norm_a_mode = norm_action_dist.mode

            exp_prob = self._get_exp_prob()
            exp_prob = torch.full([norm_a_rand.shape[0], 1], exp_prob, device=self._device, dtype=torch.float)
            rand_action_mask = torch.bernoulli(exp_prob)
            norm_a = torch.where(rand_action_mask == 1.0, norm_a_rand, norm_a_mode)
            rand_action_mask = rand_action_mask.squeeze(-1)

        elif (self._mode == base_agent.AgentMode.TEST):
            norm_a = norm_action_dist.mode
            rand_action_mask = torch.zeros_like(norm_a[..., 0])
        else:
            assert(False), "Unsupported agent mode: {}".format(self._mode)
            
        norm_a_logp = norm_action_dist.log_prob(norm_a)

        norm_a = norm_a.detach()
        norm_a_logp = norm_a_logp.detach()
        a = self._a_norm.unnormalize(norm_a)

        a_info = {
            "a_logp": norm_a_logp,
            "rand_action_mask": rand_action_mask
        }
        return a, a_info

    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("a_logp", action_info["a_logp"])
        self._exp_buffer.record("rand_action_mask", action_info["rand_action_mask"])
        return
    
    def _build_train_data(self):
        self.eval()
        
        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs}
        next_vals = torch_util.eval_minibatch(self._model.eval_critic, next_critic_inputs, self._critic_eval_batch_size)
        next_vals = next_vals.squeeze(-1).detach()

        succ_val = self._compute_succ_val()
        succ_mask = (done == base_env.DoneFlags.SUCC.value)
        next_vals[succ_mask] = succ_val

        fail_val = self._compute_fail_val()
        fail_mask = (done == base_env.DoneFlags.FAIL.value)
        next_vals[fail_mask] = fail_val

        new_vals = rl_util.compute_td_lambda_return(r, next_vals, done, self._discount, self._td_lambda)

        norm_obs = self._obs_norm.normalize(obs)
        critic_inputs = {"obs": norm_obs}
        vals = torch_util.eval_minibatch(self._model.eval_critic, critic_inputs, self._critic_eval_batch_size)
        vals = vals.squeeze(-1).detach()
        adv = new_vals - vals
        
        rand_action_mask = (rand_action_mask == 1.0).flatten()
        adv_flat = adv.flatten()
        rand_action_adv = adv_flat[rand_action_mask]
        adv_mean, adv_std = mp_util.calc_mean_std(rand_action_adv)
        norm_adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-5)
        norm_adv = torch.clamp(norm_adv, -self._norm_adv_clip, self._norm_adv_clip)
        
        self._exp_buffer.set_data("tar_val", new_vals)
        self._exp_buffer.set_data("adv", norm_adv)
        
        info = {
            "adv_mean": adv_mean,
            "adv_std": adv_std
        }
        return info
    
    def _get_exp_prob(self):
        if (np.isfinite(self._exp_anneal_samples)):
            samples = self._sample_count
            l = float(samples) / self._exp_anneal_samples
            l = np.clip(l, 0.0, 1.0)
            prob = (1.0 - l) * self._exp_prob_beg + l * self._exp_prob_end
        else:
            prob = self._exp_prob_beg
        return prob

    def _update_model(self):
        self.train()

        num_envs = self.get_num_envs()
        num_samples = self._exp_buffer.get_sample_count()
        batch_size = self._batch_size * num_envs
        num_batches = int(np.ceil(float(num_samples) / batch_size))
        train_info = dict()

        for i in range(self._update_epochs):
            for b in range(num_batches):
                batch = self._exp_buffer.sample(batch_size)
                loss_info = self._compute_loss(batch)
                loss = loss_info["loss"]
                self._optimizer.step(loss)

                torch_util.add_torch_dict(loss_info, train_info)
        
        num_steps = self._update_epochs * num_batches
        torch_util.scale_torch_dict(1.0 / num_steps, train_info)

        return train_info
    
    def _compute_loss(self, batch):
        batch["norm_obs"] = self._obs_norm.normalize(batch["obs"])
        batch["norm_action"] = self._a_norm.normalize(batch["action"])

        critic_info = self._compute_critic_loss(batch)
        actor_info = self._compute_actor_loss(batch)

        critic_loss = critic_info["critic_loss"]
        actor_loss = actor_info["actor_loss"]

        loss = actor_loss + self._critic_loss_weight * critic_loss

        info = {"loss":loss, **critic_info, **actor_info}
        return info

    def _compute_critic_loss(self, batch):
        norm_obs = batch["norm_obs"]
        tar_val = batch["tar_val"]
        pred = self._model.eval_critic(norm_obs)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    def _compute_actor_loss(self, batch):
        norm_obs = batch["norm_obs"]
        norm_a = batch["norm_action"]
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        rand_action_mask = batch["rand_action_mask"]

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]

        a_dist = self._model.eval_actor(norm_obs)
        a_logp = a_dist.log_prob(norm_a)

        a_ratio = torch.exp(a_logp - old_a_logp)
        actor_loss0 = adv * a_ratio
        actor_loss1 = adv * torch.clamp(a_ratio, 1.0 - self._ppo_clip_ratio, 1.0 + self._ppo_clip_ratio)
        actor_loss = torch.minimum(actor_loss0, actor_loss1)
        actor_loss = -torch.mean(actor_loss)
        
        clip_frac = (torch.abs(a_ratio - 1.0) > self._ppo_clip_ratio).type(torch.float)
        clip_frac = torch.mean(clip_frac)
        imp_ratio = torch.mean(a_ratio)
        
        info = {
            "actor_loss": actor_loss,
            "clip_frac": clip_frac.detach(),
            "imp_ratio": imp_ratio.detach()
        }

        if (self._action_bound_weight != 0):
            action_bound_loss = self._compute_action_bound_loss(a_dist)
            if (action_bound_loss is not None):
                action_bound_loss = torch.mean(action_bound_loss)
                actor_loss += self._action_bound_weight * action_bound_loss
                info["action_bound_loss"] = action_bound_loss.detach()

        if (self._action_entropy_weight != 0):
            action_entropy = a_dist.entropy()
            action_entropy = torch.mean(action_entropy)
            actor_loss += -self._action_entropy_weight * action_entropy
            info["action_entropy"] = action_entropy.detach()
        
        if (self._action_reg_weight != 0):
            action_reg_loss = a_dist.param_reg()
            action_reg_loss = torch.mean(action_reg_loss)
            actor_loss += self._action_reg_weight * action_reg_loss
            info["action_reg_loss"] = action_reg_loss.detach()
        
        return info

    def _log_train_info(self, train_info, test_info, env_diag_info, start_time):
        super()._log_train_info(train_info, test_info, env_diag_info, start_time)
        self._logger.log("Exp_Prob", self._get_exp_prob())
        return