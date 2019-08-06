import torch

from .attention_mlp import AttentionMLP
from .attention_mlp import Encoder
from .attention_mlp import PyTorchRandomStateContext
from .attention_mlp import F_ACTIVATIONS



class ExplicitDAGAttentionMLP(AttentionMLP):
    # TODO: Fill in documentation
    """
    """

    def __init__(
        self, n_layers: int, n_heads: int, in_features: int, attention_in_features: int, attention_hidden_features,
        attention_activation_name: str, dropout: float, reduction_name: str, use_mask: bool, mlp_extra_input_size: int,
        mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, mlp_activation_name: str, output_size: int,
        mlp_use_batch_norm: bool, mlp_use_skip: bool, *, device: str, seed: int
    ):
        super().__init__(
            n_layers, n_heads, in_features, attention_in_features, attention_hidden_features, attention_activation_name,
            dropout, reduction_name, use_mask, mlp_extra_input_size, mlp_hidden_layer_size, mlp_n_hidden_layers,
            mlp_activation_name, output_size, mlp_use_batch_norm, mlp_use_skip, device=device, seed=seed
        )

        with PyTorchRandomStateContext(seed=seed):
            self.paths_attention = Encoder(
                in_features=attention_in_features,
                hidden_features=attention_hidden_features,
                encoder_num=n_layers,
                head_num=n_heads,
                attention_activation=F_ACTIVATIONS[attention_activation_name] if attention_activation_name is not None else None,
                feed_forward_activation=F_ACTIVATIONS[mlp_activation_name],
                dropout_rate=dropout
            )
        self.paths_attention = self.paths_attention.to(device)

    def forward(self, args):
        dag_structure, dag, features = args

        # Embed the nodes in the dag
        dag = self.embedder(dag)

        for i, paths in enumerate(dag_structure):
            encoded_inputs = []
            for input_indices in paths:
                inputs = dag[:, input_indices]
                encoded_input = self._get_encoded_sequence(inputs, self.attention, self.use_mask)
                encoded_inputs.append(encoded_input.unsqueeze(dim=self.seq_len_dim))
            encoded_inputs = torch.cat(encoded_inputs, dim=self.seq_len_dim)
            encoded_node = self._get_encoded_sequence(encoded_inputs, self.paths_attention, False)
            dag[:, i] = encoded_node

        # encoded_dag = self.reduction(dag, dim=self.seq_len_dim)
        encoded_dag = dag[:, -1]
        return self._get_final_output(encoded_dag, features)
