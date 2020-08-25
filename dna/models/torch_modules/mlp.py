import typing

import torch.nn as nn

from .fully_connected_module import FullyConnectedModule


class MLP(nn.Module):

    def __init__(
            self, layer_sizes: typing.List[int], activation_name: str, use_batch_norm: bool, use_skip: bool = False,
            dropout: float = 0.0, *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__()

        self.mlp = FullyConnectedModule(
            layer_sizes, activation_name, use_batch_norm, use_skip, dropout, device=device, seed=seed
        )

    def forward(self, args):
        _, _, x = args

        return self.mlp(x).squeeze()
