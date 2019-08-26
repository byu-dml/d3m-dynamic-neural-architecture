import torch
import torch.nn as nn

from .torch_utils import PyTorchRandomStateContext


class PMF(nn.Module):
    """
    In usual Matrix Factorization terms:
    dataset_features (aka dataset_id) = movie_features
    pipeline_features (aka pipeline_id) = user_features
    Imagine it as a matrix like so:

             Datasets (id)
         ______________________
    P   | ? .2  ?  ?  .4  ?  ? |
    i   | ?  ?  ?  ?   ? .4  ? |
    p   |.7  ?  ?  ?  .8  ?  ? |
    e   | ? .1  ?  ?   ?  ?  ? |
    l   | ?  ? .9  ?   ?  0  ? |
    i   | ?  ?  ?  ?   ?  ?  ? |
    n   |.3  ?  ?  ?  .2  ?  1 |
    e   |______________________|


    """
    def __init__(self, n_pipelines, n_datasets, n_factors, device, seed):
        super(PMF, self).__init__()
        self.n_pipelines = n_pipelines
        self.n_datasets = n_datasets
        self.n_factors = n_factors
        self.device = device
        assert type(n_pipelines) == int and type(n_datasets) == int and type(n_factors) == int, "given wrong input for PMF: expected int"

        with PyTorchRandomStateContext(seed):
            self.pipeline_factors = torch.nn.Embedding(n_pipelines, n_factors, sparse=False).to(self.device)

            self.dataset_factors = torch.nn.Embedding(n_datasets, n_factors, sparse=False).to(self.device)


    def forward(self, args):
        # TODO: will using out of matmul optimize this?
        matrix = torch.matmul(
            self.pipeline_factors.weight, self.dataset_factors.weight.permute([1, 0])  # TODO: optimize by removing permute
        ).to(self.device).float()
        return matrix
