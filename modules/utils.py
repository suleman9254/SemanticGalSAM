import numpy as np
import torch
import random

def reset_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    random.seed(n)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

generator = torch.Generator()