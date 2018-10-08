import random
from typing import Type, Dict, List

import numpy as np
import pandas as pd

from data import read_data, make_cv_folds
from . import Dataset, GroupDataLoader

class BaseProblem(object):

    def __init__(
        self, data_path: str, batch_group_key: str, target_key: str,
        n_folds: int, batch_size: int, drop_last: bool, device: str,
        seed: int = None
    ):
        self.data_path = data_path
        self.batch_group_key = batch_group_key
        self.target_key = target_key
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        if seed is None:
            seed = random.randint(0, 2**32-1)
        self.seed = seed

        self._data = read_data(self.data_path)
        self._process_data()

        self._random = random.Random()
        self._random.seed(self.seed)
        self._cv_folds = make_cv_folds(
            self._data, self.batch_group_key, self.n_folds, self._random
        )
        self._fold = 0
        self._init_fold(self._fold)

    def _process_data(self):
        raise NotImplementedError()

    def _init_fold(self, fold):
        train_data = [
            self._data[i] for i in self._cv_folds[fold][0]
        ]
        test_data = [
            self._data[i] for i in self._cv_folds[fold][1]
        ]
        train_data, test_data = self._process_metafeatures(
            train_data, test_data
        )
        self._train_data_loader = GroupDataLoader(
            data = train_data,
            group_key = self.batch_group_key,
            dataset_class = Dataset,
            dataset_params = {
                "features_key": "metafeatures",
                "target_key": self.target_key,
                "device": self.device
            },
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            shuffle = True,
        )
        self._test_data_loader = GroupDataLoader(
            data = test_data,
            group_key = self.batch_group_key,
            dataset_class = Dataset,
            dataset_params = {
                "features_key": "metafeatures",
                "target_key": self.target_key,
                "device": self.device
            },
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            shuffle = True,
        )

    def _process_metafeatures(self, train_data, test_data):
        train_mfs = pd.DataFrame([item["metafeatures"] for item in train_data])
        test_mfs = pd.DataFrame([item["metafeatures"] for item in test_data])

        train_mfs.replace(
            to_replace=[np.inf, - np.inf], value=np.nan, inplace=True
        )
        drop_cols = list(train_mfs.columns[train_mfs.isnull().any()])

        train_mfs.drop(labels=drop_cols, axis=1, inplace=True)
        test_mfs.drop(labels=drop_cols, axis=1, inplace=True)

        # dev testing
        if train_mfs.shape[1] != test_mfs.shape[1]:
            raise Exception(
                "train and test metafeature size does not match: {} and {}".format(train_mfs.shape[1], test_mfs.shape[1])
            )

        mean = train_mfs.mean()
        std = train_mfs.std()

        train_mfs = (train_mfs - mean) / std
        test_mfs = (test_mfs - mean) / std

        for item, mfs in zip(train_data, train_mfs.values.tolist()):
            item["metafeatures"] = mfs

        for item, mfs in zip(test_data, test_mfs.values.tolist()):
            item["metafeatures"] = mfs

        return train_data, test_data

    def next_fold(self):
        self._fold += 1
        self._init_fold(self._fold)

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @property
    def test_data_loader(self):
        return self._test_data_loader

    @property
    def shape(self):
        return self._shape

    @property
    def loss_function(self):
        return self._loss_function

    @property
    def baseline_losses(self):
        return self._baseline_losses
