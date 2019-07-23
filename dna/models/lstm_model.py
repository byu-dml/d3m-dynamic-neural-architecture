import torch
import numpy as np

from .base_models import PyTorchRegressionRankSubsetModelBase
from .base_models import PyTorchModelBase
from .torch_modules.lstm_mlp import LSTMMLP
from dna.data import RNNDataLoader


class LSTMModel(PyTorchRegressionRankSubsetModelBase):
    def __init__(
            self, activation_name: str, hidden_state_size: int, lstm_n_layers: int, dropout: float,
            output_n_hidden_layers: int, output_hidden_layer_size: int, use_batch_norm: bool, use_skip: bool = False, *,
            device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(y_dtype=torch.float32, seed=seed, device=device)

        self.activation_name = activation_name
        self.hidden_state_size = hidden_state_size
        self.lstm_n_layers = lstm_n_layers
        self.dropout = dropout
        self.output_n_hidden_layers = output_n_hidden_layers
        self.output_hidden_layer_size = output_hidden_layer_size
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.device = device
        self.seed = seed
        self._data_loader_seed = seed + 1
        self._model_seed = seed + 2

        self.pipeline_structures = None
        self.num_primitives = None
        self.primitive_name_to_enc = None
        self.target_key = 'test_f1_macro'
        self.batch_group_key = 'pipeline_structure'
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.prim_name_key = 'name'
        self.prim_inputs_key = 'inputs'
        self.features_key = 'metafeatures'

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_ratio, patience, *,
        output_dir=None, verbose=False
    ):
        # Get the mapping of primitives to their one hot encoding
        self.primitive_name_to_enc = self._get_primitive_name_to_enc(train_data=train_data)

        PyTorchModelBase.fit(
            self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_ratio, patience,
            output_dir=output_dir, verbose=verbose
        )

    def _get_primitive_name_to_enc(self, train_data):
        primitive_names = set()

        # Get a set of all the primitives in the train set
        for instance in train_data:
            primitives = instance[self.pipeline_key][self.steps_key]
            for primitive in primitives:
                primitive_name = primitive[self.prim_name_key]
                primitive_names.add(primitive_name)

        # Get one hot encodings of all the primitives
        self.num_primitives = len(primitive_names)
        encoding = np.identity(n=self.num_primitives)

        # Create a mapping of primitive names to one hot encodings
        primitive_name_to_enc = {}
        primitive_names = sorted(primitive_names)
        for (primitive_name, primitive_encoding) in zip(primitive_names, encoding):
            primitive_name_to_enc[primitive_name] = primitive_encoding

        return primitive_name_to_enc

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
            seed=self.seed
        )

    def _get_loss_function(self):
        objective = torch.nn.MSELoss(reduction="mean")
        return lambda y_hat, y: torch.sqrt(objective(y_hat, y))

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last, shuffle=True):
        return RNNDataLoader(
            data=data,
            group_key=self.batch_group_key,
            dataset_params={
                'features_key': self.features_key,
                'target_key': self.target_key,
                'y_dtype': self.y_dtype,
                'device': self.device
            },
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=self._data_loader_seed,
            pipeline_structures=self.pipeline_structures,
            primitive_to_enc=self.primitive_name_to_enc,
            pipeline_key=self.pipeline_key,
            steps_key=self.steps_key,
            prim_name_key=self.prim_name_key
        )
