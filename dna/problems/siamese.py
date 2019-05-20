from collections import Counter
import torch

from data import TRAIN_DATA_PATH, TEST_DATA_PATH
from .base_problem import BaseProblem
from .siamese_data_loader import SiameseDataLoader
from models import Submodule, SiameseModel


class Siamese(BaseProblem):

    def __init__(
        self, train_data_path: str = TRAIN_DATA_PATH,
        test_data_path: str = TEST_DATA_PATH, n_folds: int = 5,
        batch_size = 32, drop_last = False, device = "cuda:0", seed = 0
    ):
        super(Siamese, self).__init__(
            train_data_path = train_data_path,
            test_data_path = test_data_path,
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )

        self._shape = (len(self._train_data[0][self.features_key]), 2)
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._build_models()

    def _build_models(self):
        torch_state = torch.random.get_rng_state()

        torch.manual_seed(self._randint())
        torch.cuda.manual_seed_all(self._randint())
        input_model = Submodule("input", self._shape[0], self._shape[0])
        input_model.cuda()

        submodules = {}
        for instance in self._train_data:
            for step in instance['pipeline']:
                if not step['name'] in submodules:
                    n_inputs = len(step['inputs'])
                    submodules[step['name']] = Submodule(
                        step['name'], n_inputs * self._shape[0], self._shape[0]
                    )
                    submodules[step['name']].cuda()  # todo: put on self.device

        output_model = Submodule('output', 2 * self._shape[0], self._shape[1])
        output_model.cuda()
        self._model = SiameseModel(input_model, submodules, output_model)
        if "cuda" in self.device:
            self._model.cuda()
        torch.random.set_rng_state(torch_state)

    def _get_data_loader(self, data):
        return SiameseDataLoader(data,
                                 self.batch_group_key,
                                 self.pipeline_key,
                                 self.data_set_key,
                                 self.features_key,
                                 self.target_key,
                                 self.device)

    def _compute_baselines(self):
        self._baselines = self._default_baseline()
        self._baselines.update(self._mode_baseline())

    def _default_baseline(self):
        return {
            "default": {
                "train": .5,
                "test": .5
            }
        }

    def _mode_baseline(self):
        pipeline_win_counts = {}
        for x_batch, y_batch in self._train_data_loader:
            left_pipeline, right_pipeline = x_batch[0]
            counts = Counter(y_batch.cpu().numpy())

            if not left_pipeline in pipeline_win_counts:
                pipeline_win_counts[left_pipeline] = 0
            pipeline_win_counts[left_pipeline] += counts.get(0, 0)

            if not right_pipeline in pipeline_win_counts:
                pipeline_win_counts[right_pipeline] = 0
            pipeline_win_counts[right_pipeline] = counts.get(1, 0)

        return {
            "mode_accuracy": {
                "train": self._mode_accuracy(
                    pipeline_win_counts, self._train_data_loader
                ),
                "validation_accuracy": self._mode_accuracy(
                    pipeline_win_counts, self._validation_data_loader
                )
            }
        }

    def _mode_accuracy(self, win_counts, data_loader):
        correct = 0
        total = 0
        for x_batch, y_batch in data_loader:
            left_pipeline, right_pipeline = x_batch[0]
            guess = 0
            if win_counts[left_pipeline] < win_counts[right_pipeline]:
                guess = 1
            counts = Counter(y_batch.cpu().numpy())
            correct += counts.get(guess, 0)
            total += y_batch.shape[0]
        return correct / total

    def get_correlation_coefficient(self, dataloader):
        pass



def main():
    problem = Siamese(seed = 0)
    print(problem.baselines)

if __name__ == '__main__':
    main()