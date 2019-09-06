from .base_models import RNNRegressionRankModelBase
from .torch_modules.attention_mlp import AttentionMLP


class AttentionRegressionModel(RNNRegressionRankModelBase):

    # used for plotting and reporting
    name = 'Attention'
    color = 'red'

    def __init__(
        self, n_layers: int, n_heads: int, features_per_head: int, attention_hidden_features: int,
        attention_activation_name: str, reduction_name: str, use_mask: bool, activation_name: str, dropout: float,
        output_n_hidden_layers: int, output_hidden_layer_size: int, use_batch_norm: bool, use_skip: bool,
        loss_function_name, *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(
            activation_name, dropout, output_n_hidden_layers, output_hidden_layer_size, use_batch_norm, use_skip,
            device=device, seed=seed, loss_function_name=loss_function_name
        )

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.features_per_head = features_per_head
        self.attention_hidden_features = attention_hidden_features
        self.attention_activation_name = attention_activation_name
        self.reduction_name = reduction_name
        self.use_mask = use_mask

    def _get_model(self, train_data):
        n_features = len(train_data[0][self.features_key])
        return AttentionMLP(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            in_features=self.num_primitives,
            features_per_head=self.features_per_head,
            attention_activation_name=self.attention_activation_name,
            attention_hidden_features=self.attention_hidden_features,
            dropout=self.dropout,
            reduction_name=self.reduction_name,
            use_mask=self.use_mask,
            mlp_extra_input_size=n_features,
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            mlp_activation_name=self.activation_name,
            output_size=1,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            device=self.device,
            seed=self._model_seed
        )
