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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience, save_metric, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.save_metric = save_metric
        self.delta = delta

    def compare(self, score):
        if score > self.best_score + self.delta:
            return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Saves model when the performance is better."""
        print(f"Better performance. Saving model {self.save_metric}: {self.best_score:.4f} ...")
        torch.save(model.state_dict(), self.checkpoint_path)