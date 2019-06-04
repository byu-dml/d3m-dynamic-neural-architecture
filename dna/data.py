import json
import os
import random
import tarfile
import typing

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


DATA_DIR = './data'


def group_json_objects(json_objects, group_key):
    """
    Groups JSON data by group_key.

    Parameters:
    -----------
    json_objects: List[Dict], JSON compatible list of objects.
    group_key: str, json_objects is grouped by group_key. group_key must be a
        key into each object in json_objects and the corresponding value must
        be hashable. group_key can be a '.' delimited string to access deeply
        nested fields.

    Returns:
    --------
    A dict with key being a group and the value is a list of indices into
    json_objects.
    """
    grouped_objects = {}
    for i, obj in enumerate(json_objects):
        group = obj
        for key_part in group_key.split('.'):
            group = group[key_part]
        if not group in grouped_objects:
            grouped_objects[group] = []
        grouped_objects[group].append(i)
    return grouped_objects


def split_data(data: typing.List[typing.Dict], group_by_key: str, test_size: int, seed: int):
    grouped_data_indices = group_json_objects(data, group_by_key)
    groups = list(grouped_data_indices.keys())

    rnd = random.Random()
    rnd.seed(seed)
    rnd.shuffle(groups)

    train_data = []
    for group in groups[test_size:]:
        for i in grouped_data_indices[group]:
            train_data.append(data[i])

    test_data = []
    for group in groups[:test_size]:
        for i in grouped_data_indices[group]:
            test_data.append(data[i])

    return train_data, test_data


def _extract_tarfile(path):
    assert tarfile.is_tarfile(path)

    dirname = os.path.dirname(path)
    with tarfile.open(path, 'r:*') as tar:
        members = tar.getmembers()
        if len(members) != 1:
            raise ValueError('Expected tar file with 1 member, but got {}'.format(len(members)))
        tar.extractall(os.path.dirname(path))
        extracted_path = os.path.join(dirname, tar.getmembers()[0].name)

    return extracted_path


def get_data(path):
    if tarfile.is_tarfile(path):
        path = _extract_tarfile(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


class DropMissingValues:

    def __init__(self, values_to_drop=[]):
        self.values_to_drop = values_to_drop

    def fit(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        for key, is_missing in pd.DataFrame(data).isna().any().iteritems():
            if is_missing:
                self.values_to_drop.append(key)

    def predict(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        for instance in data:
            for key in self.values_to_drop:
                instance.pop(key, None)
        return data


class StandardScaler:
    """
    Transforms data by subtracting the mean and scaling by the standard
    deviation. Drops columns that have 0 standard deviation. Clips values to
    numpy resolution, min, and max.
    """

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        values_map = {}
        for instance in data:
            for key, value in instance.items():
                if key not in values_map:
                    values_map[key] = []
                values_map[key].append(value)

        self.means = {}
        self.stds = {}
        for key, values in values_map.items():
            self.means[key] = np.mean(values)
            self.stds[key] = np.std(values, ddof=1)

    def predict(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        if self.means is None or self.stds is None:
            raise Exception('StandardScaler not fit')

        transformed_data = []
        for instance in data:
            transformed_instance = {}
            for key, value in instance.items():
                if self.stds[key] != 0:  # drop columns with 0 std dev
                    transformed_instance[key] = (value - self.means[key]) / self.stds[key]

            transformed_data.append(transformed_instance)

        return transformed_data


def preprocess_data(train_data, test_data):
    train_metafeatures = []
    for instance in train_data:
        train_metafeatures.append(instance['metafeatures'])
        for step in instance['pipeline']['steps']:
            step['name'] = step['name'].replace('.', '_')

    test_metafeatures = []
    for instance in test_data:
        test_metafeatures.append(instance['metafeatures'])
        for step in instance['pipeline']['steps']:
            step['name'] = step['name'].replace('.', '_')

    # drop metafeature if missing for any instance
    dropper = DropMissingValues()
    dropper.fit(train_metafeatures)
    train_metafeatures = dropper.predict(train_metafeatures)
    test_metafeatures = dropper.predict(test_metafeatures)

    # scale data to unit mean and unit standard deviation
    scaler = StandardScaler()
    scaler.fit(train_metafeatures)
    train_metafeatures = scaler.predict(train_metafeatures)
    test_metafeatures = scaler.predict(test_metafeatures)

    # convert from dict to list
    for instance, mf_instance in zip(train_data, train_metafeatures):
        instance['metafeatures'] = [value for key, value in sorted(mf_instance.items())]

    for instance, mf_instance in zip(test_data, test_metafeatures):
        instance['metafeatures'] = [value for key, value in sorted(mf_instance.items())]

    return train_data, test_data


class Dataset(Dataset):
    """
    A subclass of torch.utils.data.Dataset for handling simple JSON structed
    data.

    Parameters:
    -----------
    data: List[Dict], JSON structed data.
    features_key: str, the key into each element of data whose value is a list
        of features used for input to a PyTorch network.
    target_key: str, the key into each element of data whose value is the
        target used for a PyTorch network.
    device": str, the device onto which the data will be loaded
    """

    def __init__(
        self, data: typing.List[typing.Dict], features_key: str,
        target_key: str, task_type: str, device: str
    ):
        self.data = data
        self.features_key = features_key
        self.target_key = target_key
        self.task_type = task_type
        if self.task_type == "CLASSIFICATION":
            self._y_dtype = torch.int64
        elif self.task_type == "REGRESSION":
            self._y_dtype = torch.float32
        self.device = device

    def __getitem__(self, item: int):
        x = torch.tensor(
            self.data[item][self.features_key], dtype=torch.float32, device=self.device
        )
        y = torch.tensor(
            self.data[item][self.target_key],
            dtype=self._y_dtype,
            device=self.device
        )
        return x, y

    def __len__(self):
        return len(self.data)


class RandomSampler(Sampler):
    """
    Samples indices uniformly without replacement.

    Parameters
    ----------
    n: int
        the number of indices to sample
    seed: int
        used to reproduce randomization
    """

    def __init__(self, n, seed):
        self.n = n
        self._indices = list(range(n))
        self._random = random.Random()
        self._random.seed(seed)

    def __iter__(self):
        self._random.shuffle(self._indices)
        return iter(self._indices)

    def __len__(self):
        return self.n


class GroupDataLoader(object):
    """
    Batches a dataset for PyTorch Neural Network training. Partitions the
    dataset so that batches belong to the same group.

    Parameters:
    -----------
    data: List[Dict], JSON compatible list of objects representing a dataset.
        dataset_class must know how to parse the data given dataset_params.
    group_key: str, pipeline run data is grouped by group_key and each
        batch of data comes from only one group. group_key must be a key into
        each element of the pipeline run data. the value of group_key must be
        hashable.
    dataset_class: Type[torch.utils.data.Dataset], the class used to make
        dataset instances after the dataset is partitioned.
    dataset_params: dict, extra parameters needed to instantiate dataset_class
    batch_size: int, the number of data points in each batch
    drop_last: bool, default False. whether to drop the last incomplete batch.
    shuffle: bool, default True. whether to randomize the batches.
    """

    def __init__(
        self, data: typing.List[typing.Dict], group_key: str,
        dataset_class: typing.Type[Dataset], dataset_params: dict,
        batch_size: int, drop_last: bool, shuffle: bool, seed: int
    ):
        self.data = data
        self.group_key = group_key
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self._random = random.Random()
        self._random.seed(seed)
        self.old_indices = []

        self._init_dataloaders()
        self._init_group_metadataloader()

    def _init_dataloaders(self):
        """
        Groups self.data based on group_key. Creates a
        torch.utils.data.DataLoader for each group, using self.dataset_class.
        """
        # group the data
        grouped_data = group_json_objects(self.data, self.group_key)

        # create dataloaders
        self._group_dataloaders = {}
        for group, group_indices in grouped_data.items():
            self.old_indices += group_indices
            group_data = [self.data[i] for i in group_indices]
            group_dataset = self.dataset_class(group_data, **self.dataset_params)
            new_dataloader = self._get_data_loader(
                group_dataset
            )
            self._group_dataloaders[group] = new_dataloader

    def _get_data_loader(self, data):
        if self.shuffle:
            sampler = RandomSampler(len(data), self._randint())
        else:
            sampler = None
        dataloader = DataLoader(
            dataset = data,
            sampler =  sampler,
            batch_size = self.batch_size,
            drop_last = self.drop_last
        )
        return dataloader

    def _randint(self):
        return self._random.randint(0,2**32-1)

    def _init_group_metadataloader(self):
        """
        Creates a dataloader which randomizes the batches over the groups. This
        allows the order of the batches to be independent of the groups.
        """
        self._group_batches = []
        for group, group_dataloader in self._group_dataloaders.items():
            self._group_batches += [group] * len(group_dataloader)
        if self.shuffle:
            self._random.shuffle(self._group_batches)

    def unshuffle(self, predictions):
        """
        Rearrange the order from the grouping
        :param predictions:
        :return: the unshuffled predictions
        """
        # use the index order that would be sorting the array to subset
        indexes_to_order = np.argsort(np.array(self.old_indices))
        numpy_preds = predictions.numpy()
        unshuffled_preds = numpy_preds[indexes_to_order]
        return unshuffled_preds

    def __iter__(self):
        return iter(self._iter())

    def _iter(self):
        group_dataloader_iters = {}
        for group in self._group_batches:
            if not group in group_dataloader_iters:
                group_dataloader_iters[group] = iter(
                    self._group_dataloaders[group]
                )
            x_batch, y_batch = next(group_dataloader_iters[group])
            # since all pipeline are the same in this group, just grab one of them
            pipeline = self._group_dataloaders[group].dataset.data[0]["pipeline"]
            yield (group, pipeline, x_batch), y_batch
        raise StopIteration()

    def __len__(self):
        return len(self._group_batches)
