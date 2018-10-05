import random

import numpy as np
import pandas as pd

from data import TRAIN_DATA_PATH, read_data, make_cv_folds
from . import GroupDataLoader, Dataset

class Problem(object):

    def __init__(
        self, data_path=TRAIN_DATA_PATH, n_folds: int = 5, batch_size = 48,
        drop_last = False, device = "cpu", seed = None
    ):
        self.data_path = data_path
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        if seed is None:
            seed = random.randint(0, 2**32-1)
        self.seed = seed
        self._group_key = "dataset"
        self._features_key = "metafeatures"
        self._target_key = "test_accuracy"
        self._random = random.Random()
        self._random.seed(self.seed)
        self._data = read_data(TRAIN_DATA_PATH)
        self._cv_folds = make_cv_folds(
            self._data, "dataset", n_folds, self._random
        )
        self._fold_index = 0
        self._init_fold()


        self.feature_size = len(self._data[0][self._features_key])
        try:
            self.target_size = len(self._data[0][self._target_key])
        except TypeError:
            self.target_size = 1

    def _init_fold(self):
        train_data = [
            self._data[i] for i in self._cv_folds[self._fold_index][0]
        ]
        test_data = [
            self._data[i] for i in self._cv_folds[self._fold_index][1]
        ]
        train_data, test_data = self._preprocess_data(train_data, test_data)
        self.feature_size = train_data[0][self._features_key]
        self.train_data_loader = GroupDataLoader(
            data = train_data,
            group_key = self._group_key,
            dataset_class = Dataset,
            dataset_params = {
                "features_key": "metafeatures",
                "target_key": "test_accuracy",
                "device": self.device
            },
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            shuffle = True,
        )
        self.test_data_loader = GroupDataLoader(
            data = test_data,
            group_key = self._group_key,
            dataset_class = Dataset,
            dataset_params = {
                "features_key": "metafeatures",
                "target_key": "test_accuracy",
                "device": self.device
            },
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            shuffle = True,
        )

    def _preprocess_data(self, train_data, test_data):
        train_mfs = pd.DataFrame([item["metafeatures"] for item in train_data])
        test_mfs = pd.DataFrame([item["metafeatures"] for item in test_data])

        train_mfs.replace(
            to_replace=[np.inf, - np.inf], value=np.nan, inplace=True
        )
        drop_cols = list(train_mfs.columns[train_mfs.isnull().any()])

        train_mfs.drop(labels=drop_cols, axis=1, inplace=True)
        test_mfs.drop(labels=drop_cols, axis=1, inplace=True)

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
        self.fold_index += 1
        self._init_fold()
