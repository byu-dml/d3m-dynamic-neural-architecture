import typing

import torch.nn as nn

from .torch_utils import PyTorchRandomStateContext, get_activation


class FullyConnectedModule(nn.Module):
    """A torch module with fully connected layers with a non-linear activation between each layer.
     Also supports optional batch normalization and dropout between each layer and a single skip
    connection from the beginning to the end of the module.
    """

    def __init__(
        self, layer_sizes: typing.List[int], activation_name: str, use_batch_norm: bool,
        use_skip: bool = False, dropout: float = 0.0, *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__()

        with PyTorchRandomStateContext(seed):
            n_layers = len(layer_sizes) - 1
            activation = get_activation(activation_name)

            layers = []
            for i in range(n_layers):
                if i > 0:
                    layers.append(activation())
                    if dropout > 0.0:
                        layers.append(nn.Dropout(p=dropout))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i]))
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

            self.net = nn.Sequential(*layers)
            self.net.to(device=device)

            if use_skip:
                if layer_sizes[0] == layer_sizes[-1]:
                    self.skip = nn.Sequential()
                else:
                    self.skip = nn.Linear(layer_sizes[0], layer_sizes[-1])
                self.skip.to(device=device)
            else:
                self.skip = None

    def forward(self, x):
        if self.skip is None:
            return self.net(x)
        else:
            return self.net(x) + self.skip(x)
