import torch

from .base_models import PyTorchModelBase
from .base_models import PyTorchRegressionRankModelBase
from .torch_modules.torch_utils import PyTorchRandomStateContext
from .torch_modules.pmf import PMF
from dna.data import PMFDataLoader
#TODO: move all low-level torch stuff to torch_modules.pmf


class PMFLoss(torch.nn.Module):

    def __init__(self, lam_u, u_weights, lam_v, v_weights):
        super().__init__()
        self.lam_u = lam_u
        self.u_weights = u_weights
        self.lam_v = lam_v
        self.v_weights = v_weights
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, y_hat, y):
        rmse_loss = torch.sqrt(self.mse_loss(y_hat, y))

        u_loss = 0
        if self.lam_u > 0:
            u_loss = self.lam_u * torch.sum(self.u_weights.norm(dim=1))

        v_loss = 0
        if self.lam_v > 0:
            v_loss = self.lam_v * torch.sum(self.v_weights.norm(dim=1))

        return rmse_loss + u_loss + v_loss


class ProbabilisticMatrixFactorization(PyTorchRegressionRankModelBase):
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

    def __init__(self, k: int, loss_function_args, *,  device: str = 'cuda:0', seed=0):
        super().__init__(y_dtype=torch.float32, device=device, seed=seed)
        self.k = k
        self.loss_function_args = loss_function_args

    def _get_loss_function(self):
        return PMFLoss(
            self.loss_function_args.get('lam_u', 0), self.model.dataset_factors.weight,
            self.loss_function_args.get('lam_v', 0), self.model.pipeline_factors.weight
        )

    def _get_data_loader(self, data, batch_size=0, drop_last=False, shuffle=True):
        with PyTorchRandomStateContext(self.seed):
            data_loader = PMFDataLoader(
                data, self.n_pipelines, self.n_datasets, self.encode_pipeline, self.encode_dataset,
                self.pipeline_id_mapper, self.dataset_id_mapper
            )
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
            self, train_data, n_epochs, learning_rate, batch_size, False, validation_ratio, patience,
            output_dir=output_dir, verbose=verbose
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
