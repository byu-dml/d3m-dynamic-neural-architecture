import torch
import numpy as np

from data import TRAIN_DATA_PATH, TEST_DATA_PATH
from .base_problem import BaseProblem


class Regression(BaseProblem):

    def __init__(
        self, train_data_path: str = TRAIN_DATA_PATH,
        test_data_path: str = TEST_DATA_PATH, n_folds: int = 5,
        batch_size = 48, drop_last = True, device = "cuda:0", seed = 0
    ):
        self._target_key = "test_accuracy"
        objective = torch.nn.MSELoss(reduction="elementwise_mean")
        self._loss_function = lambda y, y_hat: torch.sqrt(objective(y, y_hat))
        super(Regression, self).__init__(
            train_data_path = train_data_path,
            test_data_path = test_data_path,
            batch_group_key = "dataset",
            target_key = self._target_key,
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )
        self._shape = (len(self._train_data[0]["metafeatures"]), 1)

    def _process_train_data(self):
        pass

    def _process_test_data(self):
        pass

    def _compute_baselines(self):
        train_accuracies = []
        for x_batch, y_batch in self._train_data_loader:
            train_accuracies.extend(y_batch.tolist())
        train_accuracies = np.array(train_accuracies)
        train_mean = np.mean(train_accuracies)
        train_median = np.median(train_accuracies)
        mean_rmse = np.sqrt(np.mean((train_accuracies - train_mean)**2))
        median_rmse = np.sqrt(np.mean((train_accuracies - train_median)**2))
        guess_accuracy = train_mean
        train_rmse = mean_rmse
        if median_rmse < mean_rmse:
            guess_accuracy = train_median
            train_rmse = median_rmse

        validation_accuracies = []
        for x_batch, y_batch in self._validation_data_loader:
            validation_accuracies.extend(y_batch.tolist())
        validation_accuracies = np.array(validation_accuracies)
        validation_rmse = np.sqrt(np.mean((validation_accuracies - guess_accuracy)**2))

        self._baselines = {
            "min_mean_med_rmse": {
                "train": train_rmse,
                "validation": validation_rmse
            }
        }

def main():
    problem = Regression(seed = 0)
    print(problem.baselines)

if __name__ == '__main__':
    main()
