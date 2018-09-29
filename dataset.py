import json
import random
from typing import Type, List, Dict

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch


class PipelineDataset(Dataset):

    def __init__(
        self, data: List[Dict], feature_label: str, target_label: str,
        device: str
    ):
        self.data = data
        self.feature_label = feature_label
        self.target_label = target_label
        self.device = device

    def __getitem__(self, item:int):
        x = torch.tensor(
            self.data[item][self.feature_label],
            dtype=torch.float32,
            device=self.device
        )
        y = torch.tensor(
            self.data[item][self.target_label],
            dtype=torch.float32,
            device=self.device
        )
        return x, y

    def __len__(self):
        return len(self.data)


def _preprocess_data(path):
    data = json.load(open(path, "r"))

    metafeatures = []
    for pipeline in data:
        metafeatures.append(pipeline["metafeatures"])
    mf_df = pd.DataFrame(metafeatures)
    non_time_cols = [col for col in mf_df.columns if not "time" in col.lower()]
    mf_df = mf_df[non_time_cols].replace([float("inf"), -float("inf")], float("nan"))
    mf_df = mf_df.dropna(axis=1, how="any")
    processed_mf_df = (mf_df - mf_df.mean()) / mf_df.std()

    processed_data = []
    for pipeline_data, mfs in zip(data, processed_mf_df.values.tolist()):
        dataset, pipeline = pipeline_data["job_str"].split("___", 1)
        processed_data.append({
            "dataset": dataset,
            "pipeline": pipeline,
            "metafeatures": mfs,
            "train_accuracy": pipeline_data["train_accuracy"],
            "test_accuracy": pipeline_data["test_accuracy"],
            "train_time": pipeline_data["train_fit_time"] +\
                pipeline_data["train_predict_time"],
            "test_time": pipeline_data["test_predict_time"]
        })
    json.dump(
        processed_data, open("./data/processed_data.json", "w"), indent=4
    )


class GroupDataLoader(object):
    """
    Batches a dataset for PyTorch Neural Network training. Partitions the
    dataset so that batches belong to the same group.
    Parameters:
    -----------
    path: str, path to JSON file containing pipeline run data. file must
        contain an array of objects.
    group_label: str, pipeline run data is grouped by group_label and each
        batch of data comes from only one group. group_label must be a key into
        each element of the pipeline run data. the value of group_label must be
        hashable.
    dataset_class: Type[torch.utils.data.Dataset], the class used to make
        dataset instances after the dataset is partitioned.
    dataset_params: dict, extra parameters needed to instantiate dataset_class
    batch_size: int, the number of data points in each batch
    drop_last: bool, default False. whether to drop the last incomplete batch.
    shuffle: bool, default True. whether to randomize the batches.
    seed: int, default 0. random seed.
    """

    def __init__(
        self, path: str, group_label: str, dataset_class: Type[Dataset],
        dataset_params: dict, batch_size: int, drop_last: bool = False,
        shuffle: bool = True, device: str = "cpu", seed: int = 0
    ):
        self.path = path
        self.group_label = group_label
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.device = device
        self.seed = seed

        self._load_data()
        self._init_dataloaders()
        self._init_group_metadataloader()

    def _load_data(self):
        """
        Reads the dataset from self.path.
        """
        with open(self.path, "r") as f:
            self._data = json.load(f)

    def _init_dataloaders(self):
        """
        Groups self._data based on group_label. Creates a
        torch.utils.data.DataLoader for each group, using self.dataset_class.
        """
        # group the data
        grouped_data = {}
        for data_point in self._data:
            group = data_point[self.group_label]
            if not group in grouped_data:
                grouped_data[group] = []
            grouped_data[group].append(data_point)

        # create dataloaders
        self._group_dataloaders = {}
        for group, group_data in grouped_data.items():
            group_dataset = self.dataset_class(group_data, **self.dataset_params)
            self._group_dataloaders[group] = DataLoader(
                dataset=group_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )

    def _init_group_metadataloader(self):
        group_batches = []
        for group, group_dataloader in self._group_dataloaders.items():
            group_batches += [group] * len(group_dataloader)
        self._group_metadataloader = DataLoader(
            dataset=group_batches,
            batch_size=1,
            shuffle=self.shuffle
        )

    def __iter__(self):
        group_dataloader_iters = {}
        for group in self._group_metadataloader:
            group = group[0]
            if not group in group_dataloader_iters:
                group_dataloader_iters[group] = iter(
                    self._group_dataloaders[group]
                )
            yield group, next(group_dataloader_iters[group])
        raise StopIteration()

    def __len__(self):
        return len(self._group_metadataloader)


if __name__ == '__main__':
    _preprocess_data("./data/complete_pipelines_and_metafeatures.json")
