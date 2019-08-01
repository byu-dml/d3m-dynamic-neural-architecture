import torch
from torch import nn
from torch_transformer import Encoder

from . import PyTorchRandomStateContext
from . import F_ACTIVATIONS
from .submodule import Submodule
from . import get_reduction_function


class AttentionMLP(nn.Module):
    """
    The Attention MLP applies attention to a sequence without considering the ordering of the inputs in the sequence.
    It then applies a reduction to the sequence to produce an encoded sequence as a single vector. This vector is
    concatenated to another features vector and then passed through an MLP.
    """

    def __init__(
        self, n_layers: int, n_heads: int, in_features: int, attention_in_features: int, attention_hidden_features,
        attention_activation_name: str, dropout: float, reduction_name: str, mlp_extra_input_size: int,
        mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, mlp_activation_name: str, output_size: int,
        mlp_use_batch_norm: bool, mlp_use_skip: bool, *, device: str, seed: int
    ):
        super().__init__()

        if attention_in_features % n_heads != 0:
            raise ValueError(
                'attention_in_features must be divisible by n_heads. {0} is not divisible by {1}'.format(
                    attention_in_features, n_heads
                )
            )

        # The embedder maps one hot encoded inputs to a feature space with a dimension size that the encoder can handle
        # Without the embedder, the encoder input feature size is restricted to the number of one hot encodings
        # This would be problematic because attention_in_features must be divisible by n_heads
        self.embedder = Submodule([in_features, attention_in_features], 'relu', False, False, 0, device=device, seed=seed)

        self.reduction = get_reduction_function(reduction_name)

        with PyTorchRandomStateContext(seed=seed):
            self.attention = Encoder(
                in_features=attention_in_features,
                hidden_features=attention_hidden_features,
                encoder_num=n_layers,
                head_num=n_heads,
                attention_activation=F_ACTIVATIONS[attention_activation_name] if attention_activation_name is not None else None,
                feed_forward_activation=F_ACTIVATIONS[mlp_activation_name],
                dropout_rate=dropout
            )
        self.attention = self.attention.to(device)

        mlp_input_size = attention_in_features + mlp_extra_input_size
        mlp_layer_sizes = [mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [output_size]
        self.mlp = Submodule(
            mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout, device=device, seed=seed+1
        )

    def forward(self, args):
        sequence, features = args
        embedded_sequence = self.embedder(sequence)
        attended_sequence = self.attention(embedded_sequence)

        seq_len_dim = 1
        encoded_sequence = self.reduction(attended_sequence, dim=seq_len_dim)
        mlp_input = torch.cat((encoded_sequence, features), dim=1)
        output = self.mlp(mlp_input)
        return output.squeeze()
