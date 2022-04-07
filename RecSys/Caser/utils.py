import torch
import torch.nn as nn
import numpy as np
import random
import os


activation_getter = {'iden': nn.Identity, 'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigm': nn.Sigmoid}


def set_seed(seed: int):
    # https://hoya012.github.io/blog/reproducible_pytorch/
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")