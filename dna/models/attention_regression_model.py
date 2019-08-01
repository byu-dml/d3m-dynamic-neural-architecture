from .base_models import RNNRegressionRankSubsetModelBase
from .torch_modules.attention_mlp import AttentionMLP


class AttentionRegressionModel(RNNRegressionRankSubsetModelBase):

    def __init__(
        self, n_layers: int, n_heads: int, attention_in_features: int, attention_hidden_features: int,
        attention_activation_name: str, reduction_name: str, activation_name: str, dropout: float,
        output_n_hidden_layers: int, output_hidden_layer_size: int, use_batch_norm: bool, use_skip: bool, *,
        device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(
            activation_name, dropout, output_n_hidden_layers, output_hidden_layer_size, use_batch_norm, use_skip,
            device=device, seed=seed
        )

        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attention_in_features = attention_in_features
        self.attention_hidden_features = attention_hidden_features
        self.attention_activation_name = attention_activation_name
        self.reduction_name = reduction_name

    def _get_model(self, train_data):
        n_features = len(train_data[0][self.features_key])
        return AttentionMLP(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            in_features=self.num_primitives,
            attention_in_features=self.attention_in_features,
            attention_activation_name=self.attention_activation_name,
            attention_hidden_features=self.attention_hidden_features,
            dropout=self.dropout,
            reduction_name=self.reduction_name,
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
