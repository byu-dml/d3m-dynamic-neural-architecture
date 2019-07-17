import torch
import torch.nn as nn

from . import PyTorchRandomStateContext


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
            self.pipeline_factors = torch.nn.Embedding(n_pipelines,
                                                       n_factors,
                                                       sparse=False).to(self.device)

            self.dataset_factors = torch.nn.Embedding(n_datasets,
                                                      n_factors,
                                                      sparse=False).to(self.device)

    def forward(self, args):
        pipeline_id, pipeline, x = args
        # gather embedding vectors
        pipeline_vec = pipeline["pipeline_embedding"].to(self.device)
        dataset_vec = x.long().to(self.device)
        # Find the embedding id
        dataset_nums = dataset_vec.argmax(1).tolist()
        pipeline_num = pipeline_vec.argmax(0).tolist()
        # find the matrix
        matrices = torch.matmul(self.pipeline_factors(pipeline_vec), self.dataset_factors(dataset_vec).permute([0, 2, 1])).to(self.device).float()
        # find the predicted values for each pipeline x dataset in our batch
        subset = [tuple((index, pipeline_num, dataset_nums[index])) for index in range(matrices.shape[0])]
        vecs = [matrices[sub].reshape(1) for sub in subset]
        # combined the predictions
        combined = torch.cat(vecs)
        return combined


