import torch
import torch.nn.functional as F
import torch.nn as nn


F_ACTIVATIONS = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
ACTIVATIONS = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}


class PyTorchRandomStateContext:

    def __init__(self, seed):
        self.seed = seed
        self._state = None

    def __enter__(self):
        self._state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)

    def __exit__(self, *args):
        torch.random.set_rng_state(self._state)