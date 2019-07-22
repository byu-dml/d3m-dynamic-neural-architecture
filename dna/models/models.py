import json
import os
import typing

import numpy as np
import pandas as pd
import torch

from dna import utils
from dna.data import Dataset, GroupDataLoader, PMFDataset, RNNDataLoader, group_json_objects
from .base_models import PyTorchModelBase
from .base_models import PyTorchRegressionRankSubsetModelBase
from .base_models import RegressionModelBase
from .base_models import RankModelBase
from .base_models import SubsetModelBase
from .baselines import MedianBaseline
from .baselines import MeanBaseline
from .baselines import PerPrimitiveBaseline
from .baselines import AutoSklearnMetalearner
from .baselines import LinearRegressionBaseline
from .baselines import RandomBaseline
from .baselines import MetaAutoSklearn
from .torch_modules.dna_module import DNAModule
from .torch_modules.dag_lstm_mlp import DAGLSTMMLP
from .torch_modules.hidden_mlp_dag_lstm_mlp import HiddenMLPDAGLSTMMLP
from .torch_modules.pmf import PMF


class DNARegressionModel(PyTorchRegressionRankSubsetModelBase):

    def __init__(
        self, n_hidden_layers: int, hidden_layer_size: int, activation_name: str, use_batch_norm: bool,
        use_skip: bool = False, dropout = 0.0, *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(y_dtype=torch.float32, device=device, seed=seed)

        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation_name = activation_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.output_layer_size = 1
        self._model_seed = self.seed + 1

    def _get_model(self, train_data):
        submodule_input_sizes = {}
        for instance in train_data:
            for step in instance['pipeline']['steps']:
                submodule_input_sizes[step['name']] = len(step['inputs'])
        self.input_layer_size = len(train_data[0]['metafeatures'])

        return DNAModule(
            submodule_input_sizes, self.n_hidden_layers + 1, self.input_layer_size, self.hidden_layer_size,
            self.output_layer_size, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout,
            device=self.device, seed=self._model_seed
        )

    def _get_loss_function(self):
        objective = torch.nn.MSELoss(reduction="mean")
        return lambda y_hat, y: torch.sqrt(objective(y_hat, y))

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last, shuffle=True):
        return GroupDataLoader(
            data = data,
            group_key = 'pipeline.id',
            dataset_class = Dataset,
            dataset_params = {
                'features_key': 'metafeatures',
                'target_key': 'test_f1_macro',
                'y_dtype': self.y_dtype,
                'device': self.device
            },
            batch_size = batch_size,
            drop_last = drop_last,
            shuffle = shuffle,
            seed = self.seed + 2
        )


class DAGLSTMRegressionModel(PyTorchRegressionRankSubsetModelBase):

    def __init__(
        self, activation_name: str, hidden_state_size: int, lstm_n_layers: int, dropout: float,
        output_n_hidden_layers: int, output_hidden_layer_size: int, use_batch_norm: bool, use_skip: bool = False,
        reduction: str = 'mean', *, device: str = 'cuda:0', seed: int = 0
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
        self.reduction = reduction
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
        self.primitive_name_to_enc = self._get_primitive_name_to_enc(train_data=train_data)

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
        primitive_names = sorted(primitive_names)
        for (primitive_name, primitive_encoding) in zip(primitive_names, encoding):
            primitive_name_to_enc[primitive_name] = primitive_encoding

        return primitive_name_to_enc

    def _get_model(self, train_data):
        return DAGLSTMMLP(
            lstm_input_size=self.num_primitives,
            lstm_hidden_state_size=self.hidden_state_size,
            lstm_n_layers=self.lstm_n_layers,
            dropout=self.dropout,
            mlp_extra_input_size=len(train_data[0][self.features_key]),
            mlp_hidden_layer_size=self.output_hidden_layer_size,
            mlp_n_hidden_layers=self.output_n_hidden_layers,
            output_size=1,
            mlp_activation_name=self.activation_name,
            mlp_use_batch_norm=self.use_batch_norm,
            mlp_use_skip=self.use_skip,
            reduction=self.reduction,
            device=self.device,
            seed=self._model_seed,
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
            pipeline_structures=self.pipeline_structures,
            primitive_to_enc=self.primitive_name_to_enc,
            pipeline_key=self.pipeline_key,
            steps_key=self.steps_key,
            prim_name_key=self.prim_name_key
        )

    def _get_loss_function(self):
        objective = torch.nn.MSELoss(reduction="mean")
        return lambda y_hat, y: torch.sqrt(objective(y_hat, y))


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


class ProbabilisticMatrixFactorization(PyTorchRegressionRankSubsetModelBase):
    """
    Probabilitistic Matrix Factorization (see https://arxiv.org/abs/1705.05355 for the paper)
    Adapted from traditional Probabilitistic Matrix Factorization but instead of `Users` and `Items`, we have `Pipelines` and `Datasets`
    """
    def __init__(self, latent_features, lam_u, lam_v, probabilitistic, *, device: str = 'cuda:0', seed=0):
        super().__init__(y_dtype=torch.float32, device=device, seed=seed)
        self.latent_features = latent_features
        self.device = device
        self.fitted = False

        # regularization terms to make it Probabilitistic
        self.lam_u = lam_u
        self.lam_v = lam_v
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.probabilitistic = probabilitistic

        self.target_key = 'test_f1_macro'
        self.batch_group_key = 'pipeline_structure'
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.prim_name_key = 'name'
        self.prim_inputs_key = 'inputs'
        self.features_key = 'metafeatures'

    def PMFLoss(self, y_hat, y):
        rmse_loss = torch.sqrt(self.mse_loss(y_hat, y))
        # PMF loss includes two extra regularlization
        # NOTE: using probabilitistic loss will make the loss look worse, even though it performs well on RMSE (because of the inflated)
        if self.probabilitistic:
            u_regularization = self.lam_u * torch.sum(self.model.dataset_factors.weight.norm(dim=1))
            v_regularization = self.lam_v * torch.sum(self.model.pipeline_factors.weight.norm(dim=1))
            return rmse_loss + u_regularization + v_regularization

        return rmse_loss

    def _get_loss_function(self):
        return lambda y_hat, y: self.PMFLoss(y_hat, y)

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last, shuffle=True):
        return GroupDataLoader(
            data = data,
            group_key = 'pipeline.id',
            dataset_class = PMFDataset,
            dataset_params = {
                'features_key': 'dataset_id',
                'target_key': self.target_key,
                'y_dtype': self.y_dtype,
                'device': self.device,
                "encoding_function": self.encode_dataset,
            },
            batch_size = batch_size,
            drop_last = drop_last,
            shuffle = shuffle,
            seed = self.seed + 2,
        )

    def _get_model(self, train_data):
        self.model = PMF(self.n_pipelines, self.n_datasets, self.latent_features, device=self.device, seed=self.seed)
        return self.model

    def fit(self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
        verbose=False):
        self.fitted = True

        # get mappings for matrix -> using both datasets to prepare mapping, otherwise we're unprepared for new datasets
        self.pipeline_id_mapper = self.map_pipeline_ids(train_data + validation_data)
        self.dataset_id_mapper = self.map_dataset_ids(train_data + validation_data)

        # encode the pipeline dataset mapping
        train_data = self.encode_pipeline_dataset(train_data)
        if validation_data is not None:
            validation_data = self.encode_pipeline_dataset(validation_data)

        # do the rest of the fitting
        PyTorchModelBase.fit(
            self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_data=validation_data,
            output_dir=output_dir, verbose=verbose
        )

    def encode_pipeline_dataset(self, data):
        for instance in data:
            instance["pipeline"]["pipeline_embedding"] = self.encode_pipeline(instance["pipeline"]["id"])
            instance["dataset_id_embedding"] = self.dataset_id_mapper[instance["dataset_id"]]
        return data

    def map_pipeline_ids(self, data):
        unique_pipelines = list(set([instance["pipeline"]["id"] for instance in data]))
        # for reproduciblity
        unique_pipelines.sort()
        self.n_pipelines = len(unique_pipelines)
        return {unique_pipelines[index]:index for index in range(self.n_pipelines)}

    def map_dataset_ids(self, data):
        unique_datasets = list(set([instance["dataset_id"] for instance in data]))
        unique_datasets.sort()
        self.n_datasets = len(unique_datasets)
        return {unique_datasets[index]:index for index in range(self.n_datasets)}

    def encode_dataset(self, dataset):
        dataset_vec = np.zeros(self.n_datasets)
        dataset_vec[self.dataset_id_mapper[dataset]] = 1
        dataset_vec = torch.tensor(dataset_vec.astype("int64"), device=self.device).long()
        return dataset_vec

    def encode_pipeline(self, pipeline_id):
        pipeline_vec = np.zeros(self.n_pipelines)
        pipeline_vec[self.pipeline_id_mapper[pipeline_id]] = 1
        pipeline_vec = torch.tensor(pipeline_vec.astype("int64"), device=self.device).long()
        return pipeline_vec


def get_model(model_name: str, model_config: typing.Dict, seed: int):
    model_class = {
        'dna_regression': DNARegressionModel,
        'mean_regression': MeanBaseline,
        'median_regression': MedianBaseline,
        'per_primitive_regression': PerPrimitiveBaseline,
        'autosklearn': AutoSklearnMetalearner,
        'daglstm_regression': DAGLSTMRegressionModel,
        'hidden_daglstm_regression': HiddenDAGLSTMRegressionModel,
        'linear_regression': LinearRegressionBaseline,
        'random': RandomBaseline,
        'meta_autosklearn': MetaAutoSklearn,
        'probabilistic_matrix_factorization': ProbabilisticMatrixFactorization,
    }[model_name.lower()]
    init_model_config = model_config.get('__init__', {})
    return model_class(**init_model_config, seed=seed)
