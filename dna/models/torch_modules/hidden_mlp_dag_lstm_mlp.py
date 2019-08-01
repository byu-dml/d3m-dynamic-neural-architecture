import torch.nn as nn

from . import ACTIVATIONS
from . import PyTorchRandomStateContext
from .dag_lstm import DAGLSTM
from .submodule import Submodule


class HiddenMLPDAGLSTMMLP(nn.Module):
    """
    The HiddenMLPDAGLSTMMLP combines a feature vector and a DAG using an MLP and a DAGLSTM and passes the final
    embedding to an output MLP. The feature vector is first transformed using an input MLP and is then used as the
    initial hidden state of the DAGLSTM. The DAGLSTM then creates the final embedding.
    """

    def __init__(
            self, lstm_input_size: int, lstm_hidden_state_size: int, lstm_n_layers: int, dropout: float,
            input_mlp_input_size: int, mlp_hidden_layer_size: int, mlp_n_hidden_layers: int, mlp_activation_name: str,
            output_size: int, mlp_use_batch_norm: bool, mlp_use_skip: bool, reduction_name: str, *, device: str, seed: int,
    ):
        super().__init__()

        self.device = device
        self.seed = seed
        self._input_mlp_seed = seed + 1
        self._lstm_seed = seed + 2
        self._output_mlp_seed = seed + 3

        input_mlp_layer_sizes = [input_mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [lstm_hidden_state_size]
        input_layers = [
            Submodule(
                input_mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout,
                device=self.device, seed=self._input_mlp_seed
            ),
            ACTIVATIONS[mlp_activation_name]()
        ]
        if dropout > 0.0:
            input_dropout = nn.Dropout(p=dropout)
            input_dropout.to(device=device)
            input_layers.append(input_dropout)
        if mlp_use_batch_norm:
            with PyTorchRandomStateContext(seed=seed):
                input_batch_norm = nn.BatchNorm1d(lstm_hidden_state_size)
                input_batch_norm.to(device=device)
                input_layers.append(input_batch_norm)
        self._input_mlp = nn.Sequential(*input_layers)

        self._dag_lstm = DAGLSTM(
            lstm_input_size, lstm_hidden_state_size, lstm_n_layers, dropout, reduction_name, device=self.device,
            seed=self._lstm_seed
        )
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.lstm_n_layers = lstm_n_layers

        output_mlp_input_size = lstm_hidden_state_size
        output_mlp_layer_sizes = [output_mlp_input_size] + [mlp_hidden_layer_size] * mlp_n_hidden_layers + [output_size]
        self._output_mlp = Submodule(
            output_mlp_layer_sizes, mlp_activation_name, mlp_use_batch_norm, mlp_use_skip, dropout, device=self.device,
            seed=self._output_mlp_seed,
        )

    def forward(self, args):
        (dag_structure, dags, features) = args

        batch_size = dags.shape[0]
        assert len(features) == batch_size

        lstm_start_state = self.init_hidden_and_cell_state(features)
        fc_input = self._dag_lstm(dags, dag_structure, lstm_start_state)
        return self._output_mlp(fc_input).squeeze()

    def init_hidden_and_cell_state(self, features):
        single_hidden_state = self._input_mlp(features)
        hidden_state = single_hidden_state.unsqueeze(dim=0).expand(self.lstm_n_layers, *single_hidden_state.shape)
        return (hidden_state, hidden_state)
