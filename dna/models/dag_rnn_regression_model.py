import torch
import numpy as np

from .base_models import PyTorchRegressionRankModelBase
from .base_models import PyTorchModelBase
from .base_models import RegressionModelBase
from .base_models import RankModelBase
from .data import group_json_objects
from .dag_rnn import DAGRNN
from .data import RNNDataLoader


class DAGRNNRegressionModel(PyTorchRegressionRankModelBase):

    def __init__(
            self, activation_name: str, input_n_hidden_layers: int, input_hidden_layer_size: int, input_dropout: float,
            hidden_state_size: int, lstm_n_layers: int, lstm_dropout: float, bidirectional: bool,
            output_n_hidden_layers: int, output_hidden_layer_size: int, output_dropout: float, use_batch_norm: bool,
            use_skip: bool = False, *, device: str = 'cuda:0', seed: int = 0
    ):
        PyTorchModelBase.__init__(self, y_dtype=torch.float32, seed=seed, device=device)
        RegressionModelBase.__init__(self, seed=seed)
        RankModelBase.__init__(self, seed=seed)

        self.activation_name = activation_name
        self.input_n_hidden_layers = input_n_hidden_layers
        self.input_hidden_layer_size = input_hidden_layer_size
        self.input_dropout = input_dropout
        self.hidden_state_size = hidden_state_size
        self.lstm_n_layers = lstm_n_layers
        self.lstm_dropout = lstm_dropout
        self.bidirectional = bidirectional
        self.output_n_hidden_layers = output_n_hidden_layers
        self.output_hidden_layer_size = output_hidden_layer_size
        self.output_dropout = output_dropout
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.device = device
        self.seed = seed
        self._data_loader_seed = seed + 1
        self._model_seed = seed + 2

        self.pipeline_structures = None
        self.num_primitives = None
        self.target_key = 'test_f1_macro'
        self.batch_group_key = 'pipeline_structure'
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.prim_name_key = 'name'
        self.prim_inputs_key = 'inputs'
        self.features_key = 'metafeatures'

    def fit(self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
            verbose=False):
        # Get all the pipeline structure for each pipeline structure group before encoding the pipelines
        self.pipeline_structures = {}
        grouped_by_structure = group_json_objects(train_data, self.batch_group_key)
        for (group, group_indices) in grouped_by_structure.items():
            index = group_indices[0]
            item = train_data[index]
            pipeline = item[self.pipeline_key][self.steps_key]
            group_structure = [primitive[self.prim_inputs_key] for primitive in pipeline]
            self.pipeline_structures[group] = group_structure

        # Get the mapping of primitives to their one hot encoding
        primitive_name_to_enc = self._get_primitive_name_to_enc(train_data=train_data)

        # Encode all the pipelines in the training and validation set
        self.encode_pipelines(data=train_data, primitive_name_to_enc=primitive_name_to_enc)
        if validation_data is not None:
            self.encode_pipelines(data=validation_data, primitive_name_to_enc=primitive_name_to_enc)

        PyTorchModelBase.fit(
            self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_data=validation_data,
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
        for (primitive_name, primitive_encoding) in zip(primitive_names, encoding):
            primitive_name_to_enc[primitive_name] = primitive_encoding

        return primitive_name_to_enc

    def encode_pipelines(self, data, primitive_name_to_enc):
        for instance in data:
            pipeline = instance[self.pipeline_key][self.steps_key]
            encoded_pipeline = self.encode_pipeline(pipeline=pipeline, primitive_to_enc=primitive_name_to_enc)
            instance[self.pipeline_key][self.steps_key] = encoded_pipeline

    def encode_pipeline(self, pipeline, primitive_to_enc):
        # Create a tensor of encoded primitives
        encoding = []
        for primitive in pipeline:
            primitive_name = primitive[self.prim_name_key]
            encoded_primitive = primitive_to_enc[primitive_name]
            encoding.append(encoded_primitive)
        return encoding

    def _get_model(self, train_data):
        metafeatures_length = len(train_data[0][self.features_key])
        return DAGRNN(
            activation_name=self.activation_name, input_n_hidden_layers=self.input_n_hidden_layers,
            input_hidden_layer_size=self.input_hidden_layer_size, input_dropout=self.input_dropout,
            hidden_state_size=self.hidden_state_size, lstm_n_layers=self.lstm_n_layers, lstm_dropout=self.lstm_dropout,
            bidirectional=self.bidirectional, output_n_hidden_layers=self.output_n_hidden_layers,
            output_hidden_layer_size=self.output_hidden_layer_size, output_dropout=self.output_dropout,
            use_batch_norm=self.use_batch_norm, use_skip=self.use_skip, rnn_input_size=self.num_primitives,
            input_layer_size=metafeatures_length, output_size=1, device=self.device, seed=self._model_seed
        )

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
            pipeline_structures=self.pipeline_structures
        )

    def _get_loss_function(self):
        objective = torch.nn.MSELoss(reduction="mean")
        return lambda y_hat, y: torch.sqrt(objective(y_hat, y))


