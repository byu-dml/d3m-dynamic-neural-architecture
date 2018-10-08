import torch
import numpy as np

from data import TRAIN_DATA_PATH
from .base_problem import BaseProblem


class Regression(BaseProblem):

    def __init__(
        self, data_path=TRAIN_DATA_PATH, n_folds: int = 5, batch_size = 48,
        drop_last = False, device = "cpu", seed = None
    ):
        self._target_key = "test_accuracy"
        objective = torch.nn.MSELoss(reduction="elementwise_mean")
        self._loss_function = lambda y, y_hat: torch.sqrt(objective(y, y_hat))
        super(Regression, self).__init__(
            data_path = data_path,
            batch_group_key = "dataset",
            target_key = self._target_key,
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )
        self._shape = (len(self._data[0]["metafeatures"]), 1)

    def _process_data(self):
        pass

    def _init_fold(self, fold):
        # todo min of mean and median
        # todo train and test baselines
        super(Regression, self)._init_fold(fold)
        accuracies = []
        for group, (x, y) in iter(self._train_data_loader):
            accuracies += list(y)
        accuracies = np.array(accuracies)
        mean = np.mean(accuracies)
        median = np.median(accuracies)
        mean_rmse = np.sqrt(np.mean((accuracies - mean)**2))
        median_rmse = np.sqrt(np.mean((accuracies - median)**2))
        self._baseline_losses = {
            "mean_rmse": mean_rmse,
            "median_rmse": median_rmse
        }
