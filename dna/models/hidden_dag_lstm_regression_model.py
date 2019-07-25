from .dag_lstm_regression_model import DAGLSTMRegressionModel
from .torch_modules.hidden_mlp_dag_lstm_mlp import HiddenMLPDAGLSTMMLP

class HiddenDAGLSTMRegressionModel(DAGLSTMRegressionModel):

    def __init__(
            self, activation_name: str, input_n_hidden_layers: int, input_hidden_layer_size: int, hidden_state_size: int,
            lstm_n_layers: int, dropout: float, output_n_hidden_layers: int, output_hidden_layer_size: int,
            use_batch_norm: bool, use_skip: bool = False, reduction: str = 'mean', *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(
            activation_name, hidden_state_size, lstm_n_layers, dropout, output_n_hidden_layers,
            output_hidden_layer_size, use_batch_norm, use_skip, reduction, device=device, seed=seed
        )

        self.input_n_hidden_layers = input_n_hidden_layers
        self.input_hidden_layer_size = input_hidden_layer_size

    def _get_model(self, train_data):
        n_features = len(train_data[0][self.features_key])
        return HiddenMLPDAGLSTMMLP(
            lstm_input_size=self.num_primitives,
            lstm_hidden_state_size=self.hidden_state_size,
            lstm_n_layers=self.lstm_n_layers,
            dropout=self.dropout,
            input_mlp_input_size=n_features,
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            mlp_activation_name=self.activation_name,
            output_size=1,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            reduction=self.reduction,
            device=self.device,
            seed=self._model_seed,
        )
