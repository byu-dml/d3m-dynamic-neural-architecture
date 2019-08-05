from .attention_mlp import AttentionMLP


class DAGAttentionMLP(AttentionMLP):
    """
    The DAG Attention MLP, like the Attention MLP which it extends, encodes a sequence and passes it through an MLP
    along with some additional concatenated features. But the sequence is treated like a dag, utilizing a dag structure
    passed into the forward function. The structures of the dags that this model processes not only define the immediate
    inputs to a given node, but all the inputs that come before it. This way, attention can be applied to the node
    itself and all the nodes it depends on. The structure is still ambiguous, however. The ambiguity is each node is
    defined by a collection including itself and all its dependents. But this collection does not take into account the
    various paths of dependents leading to that node, beginning at a start node.
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

    def forward(self, args):
        dag_structure, dag, features = args

        # Embed the nodes in the dag
        embedded_dag = self.embedder(dag)

        for i, inputs in enumerate(dag_structure):
            embedded_inputs = embedded_dag[:, inputs]
            encoded_input = self._get_encoded_sequence(embedded_inputs, self.attention)
            embedded_dag[:, i] = encoded_input

        encoded_dag = self.reduction(embedded_dag, dim=self.seq_len_dim)
        return self._get_final_output(encoded_dag, features)
