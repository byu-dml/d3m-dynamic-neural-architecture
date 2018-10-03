import json
import random
from typing import Type, List, Dict

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch


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


def load_data(path):
    """
    Reads the dataset from path.
    """
    return json.load(open(path, "r"))


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


class GroupDataLoader(object):
    """
    Batches a dataset for PyTorch Neural Network training. Partitions the
    dataset so that batches belong to the same group.

    Parameters:
    -----------
    data: List[Dict], JSON compatible list of objects representing a dataset.
        dataset_class must know how to parse the data given dataset_params.
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
    device: str, default "cpu". PyTorch device for the data.
    """

    def __init__(
        self, data: List[Dict], group_label: str, dataset_class: Type[Dataset],
        dataset_params: dict, batch_size: int, drop_last: bool = False,
        shuffle: bool = True, device: str = "cpu"
    ):
        self.data = data
        self.group_label = group_label
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        if batch_size == -1:
            batch_size = len(self.data)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.device = device

        self._init_dataloaders()
        self._init_group_metadataloader()

    def _init_dataloaders(self):
        """
        Groups self.data based on group_label. Creates a
        torch.utils.data.DataLoader for each group, using self.dataset_class.
        """
        # group the data
        grouped_data = self._group_data(self.data, self.group_label)

        # create dataloaders
        self._group_dataloaders = {}
        for group, group_indices in grouped_data.items():
            group_data = [self.data[i] for i in group_indices]
            group_dataset = self.dataset_class(group_data, **self.dataset_params)
            self._group_dataloaders[group] = DataLoader(
                dataset=group_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )

    def _init_group_metadataloader(self):
        """
        Creates a dataloader which randomizes the batches over the groups. This
        allows the order of the batches to be independent of the groups.
        """
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

    @classmethod
    def _group_data(cls, data, group_label):
        """
        Groups data by group_label.

        Parameters:
        -----------
        data: List[Dict], JSON compatible list of objects representing a dataset.
        group_label: str, data is grouped by group_label. group_label must be a
            key into each element of the pipeline run data. the value of
            group_label must be hashable.

        Returns:
        --------
        A dict with key being a group and the value is a list of indices into
        data.
        """
        grouped_data = {}
        for i, data_point in enumerate(data):
            group = data_point[group_label]
            if not group in grouped_data:
                grouped_data[group] = []
            grouped_data[group].append(i)
        return grouped_data

    @classmethod
    def cv_folds(
        cls, data: List[Dict], group_label: str, n_folds: int = -1,
        seed: int = 0
    ):
        grouped_data = cls._group_data(data, group_label)
        if n_folds < 0:
            n_folds = len(grouped_data)
        groups = list(grouped_data.keys())

        np.random.seed(seed)
        np.random.shuffle(groups)
        split_groups = np.array_split(groups, n_folds)

        folds = []
        for i in range(n_folds):
            train_indices = []
            test_indices = []
            for j, split_group in enumerate(split_groups):
                for group in split_group:
                    indices = grouped_data[group]
                    if i == j:
                        test_indices += indices
                    else:
                        train_indices += indices
            folds.append((train_indices, test_indices))

        return folds


if __name__ == '__main__':
    # _preprocess_data("./data/complete_pipelines_and_metafeatures.json")
    data = load_data("./data/processed_data.json")
    folds = GroupDataLoader.cv_folds(data, "dataset", 10, seed=np.random.randint(2**32))
    for fold in folds:
        print(all(i == item for i, item in enumerate(sorted(fold[0] + fold[1]))))
