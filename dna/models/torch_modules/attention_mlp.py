import torch
from torch import nn
from torch_transformer import Encoder
from torch_multi_head_attention import MultiHeadAttention

from .fully_connected_module import FullyConnectedModule
from .torch_utils import PyTorchRandomStateContext, get_activation, get_reduction


class AttentionMLP(nn.Module):
    """
    The Attention MLP applies attention to a sequence without considering the ordering of the inputs in the sequence.
    It then applies a reduction to the sequence to produce an encoded sequence as a single vector. This vector is
    concatenated to another features vector and then passed through an MLP.
    """

    def __init__(
        self, n_layers: int, n_heads: int, in_features: int, features_per_head: int, attention_hidden_features,
        attention_activation_name: str, dropout: float, reduction_name: str, use_mask: bool, mlp_extra_input_size: int,
        mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, mlp_activation_name: str, output_size: int,
        mlp_use_batch_norm: bool, mlp_use_skip: bool, *, device: str, seed: int
    ):
        super().__init__()

        self.use_mask = use_mask

        attention_in_features = features_per_head * n_heads

        self.embedder = FullyConnectedModule([in_features, attention_in_features], 'relu', False, False, 0, device=device, seed=seed)

        self.reduction = get_reduction(reduction_name)
        self.reduction_dim = 1

        with PyTorchRandomStateContext(seed=seed):
            self.attention = Encoder(
                in_features=attention_in_features,
                hidden_features=attention_hidden_features,
                encoder_num=n_layers,
                head_num=n_heads,
                attention_activation=get_activation(attention_activation_name, functional=True) if attention_activation_name is not None else None,
                feed_forward_activation=get_activation(mlp_activation_name, functional=True),
                dropout_rate=dropout
            )
        self.attention = self.attention.to(device)

        mlp_input_size = attention_in_features + mlp_extra_input_size
        mlp_layer_sizes = [mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [output_size]
        self.mlp = FullyConnectedModule(
            mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout, device=device, seed=seed+1
        )

        self.device = device

    def _get_encoded_sequence(self, sequence, attention):
        mask = None
        if self.use_mask:
            mask = MultiHeadAttention.gen_history_mask(sequence).to(self.device)

        attended_sequence = attention(sequence, mask=mask)
        encoded_sequence = self.reduction(attended_sequence, dim=self.reduction_dim)
        return encoded_sequence

    def _get_final_output(self, encoded_sequence, features):
        mlp_input = torch.cat((encoded_sequence, features), dim=1)
        output = self.mlp(mlp_input)
        return output.squeeze()

    def forward(self, args):
        sequence, features = args
        embedded_sequence = self.embedder(sequence)
        encoded_sequence = self._get_encoded_sequence(embedded_sequence, self.attention)
        return self._get_final_output(encoded_sequence, features)
