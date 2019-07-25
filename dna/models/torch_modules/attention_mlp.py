import torch
from torch import nn
from transformer import Encoder

from .submodule import Submodule
from dna.utils import get_reduction_function


class AttentionMLP(nn.Module):
    """
    The Attention MLP applies attention to a sequence without considering the ordering of the inputs in the sequence.
    It then applies a reduction to the sequence to produce an encoded sequence as a single vector. This vector is
    concatenated to another features vector and then passed through an MLP.
    """

    def __init__(
            self, n_layers: int, n_heads: int, dim_model: int, dim_keys, dim_values, dropout: float, reduction: str,
            mlp_extra_input_size: int, mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, mlp_activation_name: str,
            output_size: int, mlp_use_batch_norm: bool, mlp_use_skip: bool, *, device: str, seed: int
    ):

        super().__init__()

        self.reduction = get_reduction_function(reduction)
        dimensions = (dim_model, dim_keys, dim_values)
        self.attention = Encoder(n_layers, n_heads, *dimensions, dropout, dropout, 0)

        mlp_input_size = dim_model + mlp_extra_input_size
        mlp_layer_sizes = [mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [output_size]
        self.mlp = Submodule(mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout,
                             device=device, seed=seed+1)

    def forward(self, args):
        sequence, features = args
        sequence = self.attention(sequence)
        encoded_sequence = self.reduction(
            torch.stack(sequence, dim=0)
        )
        mlp_input = torch.cat((encoded_sequence, features), dim=1)
        output = self.mlp(mlp_input)
        return output.squeeze()