import numpy as np
import torch

from .base_models import RNNRegressionRankSubsetModelBase
from .torch_modules.lstm_mlp import LSTMMLP
from dna.data import RNNDataLoader


class LSTMModel(RNNRegressionRankSubsetModelBase):
    def __init__(
            self, activation_name: str, hidden_state_size: int, lstm_n_layers: int, dropout: float,
            output_n_hidden_layers: int, output_hidden_layer_size: int, use_batch_norm: bool, use_skip: bool = False, *,
            device: str = 'cuda:0', seed: int = 0
    ):

        super().__init__(device=device, seed=seed)

        self.activation_name = activation_name
        self.hidden_state_size = hidden_state_size
        self.lstm_n_layers = lstm_n_layers
        self.dropout = dropout
        self.output_n_hidden_layers = output_n_hidden_layers
        self.output_hidden_layer_size = output_hidden_layer_size
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip

    def _get_model(self, train_data):
        n_features = len(train_data[0][self.features_key])
        return LSTMMLP(
            input_size=self.num_primitives,
            hidden_size=self.hidden_state_size,
            lstm_n_layers=self.lstm_n_layers,
            dropout=self.dropout,
            mlp_extra_input_size=n_features,
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            output_size=1,
            mlp_activation_name=self.activation_name,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            device=self.device,
            seed=self._model_seed)
