import random
import numpy as np
import torch

def set_rand_seed(seed):
    int_seed = int(seed)
    random.seed(int_seed)
    np.random.seed(np.uint64(seed % (2**32)))
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return