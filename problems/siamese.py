import torch
import numpy as np

from data import group_json_objects, TRAIN_DATA_PATH
from .base_problem import BaseProblem
from models import PrimitiveModel, ClassificationModel, SiameseModel

class Siamese(BaseProblem):

    def __init__(
        self, data_path=TRAIN_DATA_PATH, n_folds: int = 5, batch_size = 48,
        drop_last = False, device = "cpu", seed = None
    ):
        self._target_key = "target"
        super(Siamese, self).__init__(
            data_path = data_path,
            batch_group_key = "pipelines",
            target_key = self._target_key,
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )
        self._shape = (len(self._data[0]["metafeatures"]), 2)
        self._loss_function = torch.nn.CrossEntropyLoss()
        self._baseline_losses = {"cross_entropy": 0.5}
        self._build_models()

    def _process_data(self):
        data_grouped_by_dataset = group_json_objects(self._data, "dataset")
        processed_data = []
        for group, grouped_data_indices in data_grouped_by_dataset.items():
            grouped_data = np.array(self._data)[grouped_data_indices] # todo optimize
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
        self._data = processed_data

    def _build_models(self):
        input_model = PrimitiveModel("input", self._shape[0])
        submodels = {}
        for item in self._data:
            primitive_names = item["pipelines"][0].split("___")
            for primitive_name in primitive_names:
                if not primitive_name in submodels:
                    submodels[primitive_name] = PrimitiveModel(
                        primitive_name, self.shape[0]
                    )
        output_model = ClassificationModel(2 * self._shape[0], self._shape[1])
        self._model = SiameseModel(input_model, submodels, output_model)

    @property
    def model(self):
        return self._model
    