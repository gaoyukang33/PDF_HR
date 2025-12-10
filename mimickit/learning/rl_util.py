import torch

import envs.base_env as base_env

def compute_td_lambda_return(r, next_vals, done, discount, td_lambda):
    assert(r.shape == next_vals.shape)

    return_t = torch.zeros_like(r)
    reset_mask = done != base_env.DoneFlags.NULL.value
    reset_mask = reset_mask.type(torch.float)

    last_val = r[-1] + discount * next_vals[-1]
    return_t[-1] = last_val

    timesteps = r.shape[0]
    for i in reversed(range(0, timesteps - 1)):
        curr_r = r[i]
        curr_reset = reset_mask[i]
        next_v = next_vals[i]
        next_ret = return_t[i + 1]

        curr_lambda = td_lambda * (1.0 - curr_reset)
        curr_val = curr_r + discount * ((1.0 - curr_lambda) * next_v + curr_lambda * next_ret)
        return_t[i] = curr_val

    return return_t
