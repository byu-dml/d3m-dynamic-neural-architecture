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

    test_metafeatures = []
    for instance in test_data:
        test_metafeatures.append(instance['metafeatures'])

    # drop metafeature if missing for any instance
    dropper = DropMissingValues(['pca_determinant_of_covariance'])
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
