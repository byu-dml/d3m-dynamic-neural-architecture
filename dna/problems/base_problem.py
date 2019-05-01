import copy
import json
import os
import random
from typing import Type, Dict, List

import numpy as np
import pandas as pd

from data import read_json, write_json, make_cv_folds
from . import Dataset, GroupDataLoader


class BaseProblem(object):

    def __init__(
        self, train_data_path: str, test_data_path: str, batch_group_key: str,
        target_key: str, task_type: str, n_folds: int, batch_size: int,
        drop_last: bool, device: str, seed: int
    ):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.batch_group_key = batch_group_key
        self.target_key = target_key
        self.task_type = task_type
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device
        self.seed = seed
        self._random = random.Random()
        self._random.seed(seed)

        self._load_data()
        self._cv_folds = make_cv_folds(
            self._train_data,
            self.batch_group_key,
            self.n_folds,
            self._randint()
        )
        self._fold = 0
        self._init_fold(self._fold)

    def _load_data(self):
        self._train_data = read_json(self.train_data_path)
        self._process_train_data()

        self._test_data = read_json(self.test_data_path)
        self._process_test_data()

    def _process_train_data(self):
        raise NotImplementedError()

    def _process_test_data(self):
        raise NotImplementedError()

    def _init_fold(self, fold):
        self.train_data = [
            self._train_data[i] for i in self._cv_folds[fold][0]
        ]
        self.validation_data = [
            self._train_data[i] for i in self._cv_folds[fold][1]
        ]
        self._train_data_loader = self._get_data_loader(self.train_data)
        self._validation_data_loader = self._get_data_loader(self.validation_data)
        self._compute_baselines()

    def _compute_baselines(self):
        self._baselines = {
            "default": {
                "train": np.nan,
                "validation": np.nan,
                "test": np.nan
            }
        }

    def _get_data_loader(self, data):
        return GroupDataLoader(
            data = data,
            group_key = self.batch_group_key,
            dataset_class = Dataset,
            dataset_params = {
                "features_key": "metafeatures",
                "target_key": self.target_key,
                "task_type": self.task_type,
                "device": self.device
            },
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            shuffle = True,
            seed = self._randint()
        )

    def next_fold(self):
        self._fold += 1
        self._init_fold(self._fold)

    @property
    def train_data_loader(self):
        return self._train_data_loader

    @property
    def validation_data_loader(self):
        return self._validation_data_loader

    @property
    def test_data_loader(self):
        return self._get_data_loader(self._test_data)

    @property
    def loss_function(self):
        return self._loss_function

    @property
    def baselines(self):
        return self._baselines

    @property
    def model(self):
        return self._model

    def _randint(self):
        return self._random.randint(0, 2**32-1)

    def get_correlation_coefficient(self, dataloader):
        raise NotImplementedError()
