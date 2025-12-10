import torch

import learning.base_agent as base_agent
import learning.amp_agent as amp_agent
import learning.ase_model as ase_model
import learning.rl_util as rl_util
import util.mp_util as mp_util
import util.torch_util as torch_util
import envs.base_env as base_env

import learning.distribution_gaussian_diag as distribution_gaussian_diag

class ASEAgent(amp_agent.AMPAgent):
    NAME = "ASE"

    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        self._build_latent_buf()
        return
    
    def _load_params(self, config):
        super()._load_params(config)
        
        self._latent_time_min = config["latent_time_min"]
        self._latent_time_max = config["latent_time_max"]

        self._diversity_weight = config["diversity_weight"]
        self._diversity_tar = config["diversity_tar"]
        
        self._enc_loss_weight = config["enc_loss_weight"]
        self._enc_eval_batch_size = int(config.get("enc_eval_batch_size", 0))
        self._enc_reward_weight = config["enc_reward_weight"]

        return
    
    def _build_model(self, config):
        model_config = config["model"]
        self._model = ase_model.ASEModel(model_config, self._env)
        return

    def _build_latent_buf(self):
        num_envs = self.get_num_envs()
        z_dim = self._get_latent_dim()

        self._latent_buf = torch.zeros([num_envs, z_dim], dtype=torch.float32, device=self._device)
        self._latent_reset_time = torch.zeros([num_envs], dtype=torch.float32, device=self._device)
        return
    
    def _get_latent_dim(self):
        return self._model.get_latent_dim()

    def _init_train(self):
        super()._init_train()
        self._reset_latents()
        return
    
    def _reset_envs(self, env_ids=None):
        obs, info = super()._reset_envs(env_ids)
        self._reset_latents(env_ids)
        return obs, info

    def _decide_action(self, obs, info):
        norm_obs = self._obs_norm.normalize(obs)
        z = self._latent_buf
        norm_action_dist = self._model.eval_actor(norm_obs, z)

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
    
    def _reset_latents(self, env_ids=None):
        if (env_ids is None):
            num_envs = self.get_num_envs()
            env_ids = torch.arange(num_envs, device=self._device, dtype=torch.long)

        if (len(env_ids) > 0):
            n = len(env_ids)
            z = self._sample_latents(n)
            self._latent_buf[env_ids] = z
        
            t = torch.rand(n, device=self._device)
            dt = (self._latent_time_max - self._latent_time_min) * t + self._latent_time_min
            curr_time = self._env.get_env_time(env_ids)
            rand_time = curr_time + dt
            self._latent_reset_time[env_ids] = rand_time

        return
    
    def _step_env(self, action):
        obs, r, done, info = super()._step_env(action)
        self._update_latents()
        return obs, r, done, info

    def _update_latents(self):
        curr_time = self._env.get_env_time()
        need_reset = curr_time >= self._latent_reset_time
        
        if (torch.any(need_reset)):
            env_ids = need_reset.nonzero(as_tuple=False)
            env_ids = torch.flatten(env_ids)
            self._reset_latents(env_ids)
        return

    def _sample_latents(self, n):
        z_dim = self._get_latent_dim()
        unorm_z = torch.normal(torch.zeros([n, z_dim], device=self._device))
        z = torch.nn.functional.normalize(unorm_z, dim=-1)
        return z
    
    def _record_data_pre_step(self, obs, info, action, action_info):
        super()._record_data_pre_step(obs, info, action, action_info)
        self._exp_buffer.record("latents", self._latent_buf)
        return
    
    def _build_train_data(self):
        self.eval()
        
        self._record_disc_demo_data()
        info = self._compute_rewards()

        obs = self._exp_buffer.get_data("obs")
        next_obs = self._exp_buffer.get_data("next_obs")
        r = self._exp_buffer.get_data("reward")
        done = self._exp_buffer.get_data("done")
        latents = self._exp_buffer.get_data("latents")
        rand_action_mask = self._exp_buffer.get_data("rand_action_mask")
        
        norm_next_obs = self._obs_norm.normalize(next_obs)
        next_critic_inputs = {"obs": norm_next_obs, "z": latents}
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
        critic_inputs = {"obs": norm_obs, "z": latents}
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
        
        info["adv_mean"] = adv_mean
        info["adv_std"] = adv_std

        return info

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        latents = self._exp_buffer.get_data_flat("latents")

        disc_r = self._calc_disc_rewards(norm_disc_obs)
        enc_r = self._calc_enc_rewards(tar_latents=latents, norm_enc_obs=norm_disc_obs)

        r = self._task_reward_weight * task_r \
            + self._disc_reward_weight * disc_r \
            + self._enc_reward_weight * enc_r

        self._exp_buffer.set_data_flat("reward", r)
        
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)
        enc_reward_std, enc_reward_mean = torch.std_mean(enc_r)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "enc_reward_mean": enc_reward_mean,
            "enc_reward_std": enc_reward_std
        }
        return info

    def _calc_enc_rewards(self, tar_latents, norm_enc_obs):
        with torch.no_grad():
            enc_inputs = {"enc_obs": norm_enc_obs}
            enc_pred = torch_util.eval_minibatch(self._model.eval_enc, enc_inputs, self._enc_eval_batch_size)
            err = self._calc_enc_error(tar_latents=tar_latents, enc_pred=enc_pred)
            enc_r = torch.clamp_min(-err, 0.0)
        return enc_r

    def _compute_actor_loss(self, batch):
        norm_obs = batch["norm_obs"]
        norm_a = batch["norm_action"]
        old_a_logp = batch["a_logp"]
        adv = batch["adv"]
        rand_action_mask = batch["rand_action_mask"]
        latents = batch["latents"]

        # loss should only be computed using samples with random actions
        rand_action_mask = (rand_action_mask == 1.0)
        norm_obs = norm_obs[rand_action_mask]
        norm_a = norm_a[rand_action_mask]
        old_a_logp = old_a_logp[rand_action_mask]
        adv = adv[rand_action_mask]
        latents = latents[rand_action_mask]

        a_dist = self._model.eval_actor(norm_obs, latents)
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

        if (self._diversity_weight != 0):
            diversity_loss = self._compute_diversity_loss(norm_obs, a_dist, latents)
            diversity_loss = torch.mean(diversity_loss)
            actor_loss += self._diversity_weight * diversity_loss
            info["diversity_loss"] = diversity_loss.detach()
        
        return info
    
    def _compute_loss(self, batch):
        info = super()._compute_loss(batch)

        enc_loss = self._compute_enc_loss(batch)

        loss = info["loss"]
        loss = loss + self._enc_loss_weight * enc_loss
        info["loss"] = loss
        info["enc_loss"] = enc_loss

        return info

    def _compute_critic_loss(self, batch):
        norm_obs = batch["norm_obs"]
        tar_val = batch["tar_val"]
        z = batch["latents"]
        pred = self._model.eval_critic(obs=norm_obs, z=z)
        pred = pred.squeeze(-1)

        diff = tar_val - pred
        loss = torch.mean(torch.square(diff))

        info = {
            "critic_loss": loss
        }
        return info

    def _compute_enc_loss(self, batch):
        disc_obs = batch["disc_obs"]
        tar_latents = batch["latents"]
        disc_obs = disc_obs[:self._disc_batch_size]
        tar_latents = tar_latents[:self._disc_batch_size]

        norm_disc_obs = self._disc_obs_norm.normalize(disc_obs)
        enc_pred = self._model.eval_enc(norm_disc_obs)
        enc_err = self._calc_enc_error(tar_latents=tar_latents, enc_pred=enc_pred)
        enc_loss = torch.mean(enc_err)

        return enc_loss

    def _calc_enc_error(self, tar_latents, enc_pred):
        err = tar_latents * enc_pred
        err = -torch.sum(err, dim=-1)
        return err

    def _compute_diversity_loss(self, norm_obs, action_dist, latents):
        assert(isinstance(action_dist, distribution_gaussian_diag.DistributionGaussianDiag))

        n = norm_obs.shape[0]
        new_z = self._sample_latents(n)
        new_a_dist = self._model.eval_actor(norm_obs, new_z)

        a_diff = new_a_dist.mean - action_dist.mean
        a_diff = torch.mean(torch.square(a_diff), dim=-1)

        z_diff = new_z * latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_ratio = a_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(self._diversity_tar - diversity_ratio)

        return diversity_loss
