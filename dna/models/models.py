import typing

import numpy as np
import torch

from .base_models import PyTorchModelBase, PyTorchRegressionRankSubsetModelBase
from .baselines import (
    AutoSklearnMetalearner, LinearRegressionBaseline, MeanBaseline, MedianBaseline, MetaAutoSklearn,
    PerPrimitiveBaseline, RandomBaseline
)
from .lstm_model import LSTMModel
from .torch_modules import PyTorchRandomStateContext
from .torch_modules.dag_lstm_mlp import DAGLSTMMLP
from .torch_modules.dna_module import DNAModule
from .torch_modules.hidden_mlp_dag_lstm_mlp import HiddenMLPDAGLSTMMLP
from .torch_modules.pmf import PMF
from dna.data import Dataset, GroupDataLoader, PMFDataLoader, group_json_objects


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
        objective = torch.nn.MSELoss(reduction='mean')
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


class DAGLSTMRegressionModel(LSTMModel):

    def __init__(
        self, activation_name: str, hidden_state_size: int, lstm_n_layers: int, dropout: float,
        output_n_hidden_layers: int, output_hidden_layer_size: int, use_batch_norm: bool, use_skip: bool = False,
        reduction: str = 'mean', *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(
            activation_name, hidden_state_size, lstm_n_layers, dropout, output_n_hidden_layers,
            output_hidden_layer_size, use_batch_norm, use_skip=use_skip, device=device, seed=seed
        )

        self.reduction = reduction

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_ratio, patience, *, output_dir=None,
        verbose=False
    ):
        # Get all the pipeline structure for each pipeline structure group before encoding the pipelines
        self.pipeline_structures = {}
        grouped_by_structure = group_json_objects(train_data, self.batch_group_key)
        for (group, group_indices) in grouped_by_structure.items():
            index = group_indices[0]
            item = train_data[index]
            pipeline = item[self.pipeline_key][self.steps_key]
            group_structure = [primitive[self.prim_inputs_key] for primitive in pipeline]
            self.pipeline_structures[group] = group_structure

        super().fit(
            train_data, n_epochs, learning_rate, batch_size, drop_last, validation_ratio, patience,
            output_dir=output_dir, verbose=verbose
        )

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

    Parameters
    ----------
    k: int
        the number of latent features
    probabilistic: bool
        whether to use the probabilistic component in the matrix factorization
    lam_u: float
        a regularization term used when probabilistic is True
    lam_v: float
        a regularization term used when probabilistic is True
    """
    def __init__(self, k: int, probabilitistic: bool, lam_u: float, lam_v: float, *, device: str = 'cuda:0', seed=0):
        super().__init__(y_dtype=torch.float32, device=device, seed=seed)
        self.k = k
        self.probabilitistic = probabilitistic
        self.lam_u = lam_u
        self.lam_v = lam_v

        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    # TODO: can we optimize the loss function by not initializing every call?
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
        return self.PMFLoss

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size=0, drop_last=False, shuffle=True):
        with PyTorchRandomStateContext(self.seed):
            data_loader = PMFDataLoader(data, self.n_pipelines, self.n_datasets, self.encode_pipeline, self.encode_dataset, self.pipeline_id_mapper,
                                        self.dataset_id_mapper)
            assert len(data_loader) == 1, 'PMF dataloader should have a size of 1 not {}'.format(len(data_loader))
            return data_loader

    def _get_model(self, train_data):
        self.model = PMF(self.n_pipelines, self.n_datasets, self.k, device=self.device, seed=self.seed)
        return self.model

    def fit(
        self, train_data, n_epochs, learning_rate, validation_ratio, patience, *, output_dir=None, verbose=False
    ):
        batch_size = 0

        # get mappings for matrix -> using both datasets to prepare mapping, otherwise we're unprepared for new datasets
        self.pipeline_id_mapper = self.map_pipeline_ids(train_data)
        self.dataset_id_mapper = self.map_dataset_ids(train_data)

        # do the rest of the fitting
        PyTorchModelBase.fit(
            self, train_data, n_epochs, learning_rate, batch_size, False, validation_ratio, patience, output_dir=output_dir,
            verbose=verbose
        )

    def map_pipeline_ids(self, data):
        unique_pipelines = list(set([instance['pipeline_id'] for instance in data]))
        # for reproduciblity
        unique_pipelines.sort()
        self.n_pipelines = len(unique_pipelines)
        return {unique_pipelines[index]:index for index in range(self.n_pipelines)}

    def map_dataset_ids(self, data):
        unique_datasets = list(set([instance['dataset_id'] for instance in data]))
        unique_datasets.sort()
        self.n_datasets = len(unique_datasets)
        return {unique_datasets[index]:index for index in range(self.n_datasets)}

    def encode_dataset(self, dataset):
        dataset_vec = np.zeros(self.n_datasets)
        dataset_vec[self.dataset_id_mapper[dataset]] = 1
        dataset_vec = torch.tensor(dataset_vec.astype('int64'), device=self.device).long()
        return dataset_vec

    def encode_pipeline(self, pipeline_id):
        try:
            return self.pipeline_id_mapper[pipeline_id]
        except KeyError as e:
            raise KeyError('Pipeline ID was not in the mapper. Perhaps the pipeline id was not in the training set?')

    def predict_regression(self, data, *, verbose, **kwargs):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, drop_last=False, shuffle=False)
        prediction_matrix, target_matrix = self._predict_epoch(data_loader, self._model, verbose=verbose)
        predictions = data_loader.get_predictions_from_matrix(data, prediction_matrix)
        return predictions

    def predict_rank(self, data, *, verbose, **kwargs):
        # no batch size needed
        return super().predict_rank(data, batch_size=0, verbose=verbose)

    def predict_subset(self, data, k, *, verbose=False):
        return super().predict_subset(data, k, batch_size=0, verbose=verbose)


def get_model(model_name: str, model_config: typing.Dict, seed: int):
    model_class = {
        'dna_regression': DNARegressionModel,
        'mean_regression': MeanBaseline,
        'median_regression': MedianBaseline,
        'per_primitive_regression': PerPrimitiveBaseline,
        'autosklearn': AutoSklearnMetalearner,
        'lstm': LSTMModel,
        'daglstm_regression': DAGLSTMRegressionModel,
        'hidden_daglstm_regression': HiddenDAGLSTMRegressionModel,
        'linear_regression': LinearRegressionBaseline,
        'random': RandomBaseline,
        'meta_autosklearn': MetaAutoSklearn,
        'probabilistic_matrix_factorization': ProbabilisticMatrixFactorization,
    }[model_name.lower()]
    init_model_config = model_config.get('__init__', {})
    return model_class(**init_model_config, seed=seed)
