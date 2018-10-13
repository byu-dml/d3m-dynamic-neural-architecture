from collections import Counter

import numpy as np
import torch
from tqdm import tqdm

from data import group_json_objects, TRAIN_DATA_PATH, TEST_DATA_PATH
from .base_problem import BaseProblem
from models import PrimitiveModel, ClassificationModel, SiameseModel

class Siamese(BaseProblem):

    def __init__(
        self, train_data_path: str = TRAIN_DATA_PATH,
        test_data_path: str = TEST_DATA_PATH, n_folds: int = 5,
        batch_size = 48, drop_last = True, device = "cuda:0", seed = 0
    ):
        self._target_key = "target"
        super(Siamese, self).__init__(
            train_data_path = train_data_path,
            test_data_path = test_data_path,
            batch_group_key = "pipelines",
            target_key = self._target_key,
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )
        self._shape = (len(self._train_data[0]["metafeatures"]), 2)
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._build_models()

    def _process_train_data(self):
        self._train_data = self._process_data(self._train_data)

    def _process_test_data(self):
        self._test_data = self._process_data(self._test_data)

    def _process_data(self, data):
        data_grouped_by_dataset = group_json_objects(data, "dataset")
        processed_data = []
        for group, grouped_data_indices in data_grouped_by_dataset.items():
            grouped_data = np.array(data)[grouped_data_indices] # todo optimize
            for i, data_point_i in enumerate(grouped_data):
                dataset = data_point_i["dataset"]
                mfs = data_point_i["metafeatures"]
                for j, data_point_j in enumerate(grouped_data):

                    # dev testing
                    if dataset != data_point_j["dataset"] or mfs != data_point_j["metafeatures"]:
                        raise ValueError("metafeatures are not equal")

                    if data_point_i["test_accuracy"] > data_point_j["test_accuracy"]:
                        target = 0
                    elif data_point_i["test_accuracy"] < data_point_j["test_accuracy"]:
                        target = 1
                    else:
                        continue

                    processed_data.append({
                        "dataset": dataset,
                        "metafeatures": mfs,
                        "pipelines": (
                            data_point_i["pipeline"], data_point_j["pipeline"]
                        ),
                        self._target_key: target
                    })
        return processed_data

    def _build_models(self):
        torch_state = torch.random.get_rng_state()

        torch.manual_seed(self._randint())
        torch.cuda.manual_seed_all(self._randint())
        input_model = PrimitiveModel("input", self._shape[0])
        submodels = {}
        for item in self._train_data:
            primitive_names = item["pipelines"][0].split("___")
            for primitive_name in primitive_names:
                if not primitive_name in submodels:
                    submodels[primitive_name] = PrimitiveModel(
                        primitive_name, self._shape[0]
                    )
        output_model = ClassificationModel(2 * self._shape[0], self._shape[1])
        self._model = SiameseModel(input_model, submodels, output_model)
        if "cuda" in self.device:
            self._model.cuda()
        torch.random.set_rng_state(torch_state)

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


def main():
    problem = Siamese(seed = 0)
    print(problem.baselines)

if __name__ == '__main__':
    main()