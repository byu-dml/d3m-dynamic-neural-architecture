import torch
import numpy as np

from data import TRAIN_DATA_PATH, TEST_DATA_PATH
from .base_problem import BaseProblem
from models import Submodule, DNAModel


class Regression(BaseProblem):

    def __init__(
        self, train_data_path: str = TRAIN_DATA_PATH,
        test_data_path: str = TEST_DATA_PATH, n_folds: int = 5,
        batch_size = 32, drop_last = False, device = "cuda:0", seed = 0
    ):
        self._target_key = "test_accuracy"
        objective = torch.nn.MSELoss(reduction="mean")
        self._loss_function = lambda y, y_hat: torch.sqrt(objective(y, y_hat))
        super(Regression, self).__init__(
            train_data_path = train_data_path,
            test_data_path = test_data_path,
            batch_group_key = "pipeline_id",
            target_key = self._target_key,
            task_type = "REGRESSION",
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )
        self._shape = (len(self._train_data[0]["metafeatures"]), 1)
        self._init_model()

    def _init_model(self):
        torch_state = torch.random.get_rng_state()

        torch.manual_seed(self._randint())
        torch.cuda.manual_seed_all(self._randint())
        input_model = Submodule(self._shape[0], self._shape[0])
        input_model.cuda()

        submodules = {}
        for instance in self._train_data:
            for step in instance['pipeline']:
                if not step['name'] in submodules:
                    n_inputs = len(step['inputs'])
                    submodules[step['name']] = Submodule(
                        n_inputs * self._shape[0], self._shape[0]
                    )
                    submodules[step['name']].cuda()  # todo: put on self.device

        output_model = Submodule(self._shape[0], 1, use_skip=False)
        output_model.cuda()
        self._model = DNAModel(input_model, submodules, output_model)
        if "cuda" in self.device:
            self._model.cuda()
        torch.random.set_rng_state(torch_state)

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
