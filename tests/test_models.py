import unittest

import argparse
import json
import os

from dna.__main__ import configure_evaluate_parser, evaluate, get_train_and_test_data
from dna.models.models import get_model
from dna.problems import get_problem
from dna.data import _extract_tarfile


class ModelDeterminismTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_path = './data/small_classification_train.json'
        cls.raw_data_path =  './data/small_classification_train.tar.xz'
        if not os.path.isfile(cls.data_path):
            _extract_tarfile(self.raw_data_path))

    def test_dna_regression_determinism(self):
        self._test_determinism(
            model='dna_regression', model_config_path='./model_configs/dna_regression_config.json'
        )

    def test_dag_lstm_regression_determinism(self):
        self._test_determinism(
            model='daglstm_regression', model_config_path='./model_configs/daglstm_regression_config.json'
        )

    def _test_determinism(self, model: str, model_config_path: str):
        # Set the arguments for this test
        parser = argparse.ArgumentParser()
        configure_evaluate_parser(parser)
        argv = [
            '--model', model,
            '--model-config-path', model_config_path,
            '--model-seed', '0',
            '--problem', 'regression', 'rank', 'subset',
            '--k', '2',
            '--train-path', self.data_path,
            '--test-size', '2',
            '--split-seed', '0',
            '--metafeature-subset', 'all',
            '--no-cache',
        ]
        arguments = parser.parse_args(argv)

        results1 = self._evaluate_model(arguments)
        results2 = self._evaluate_model(arguments)
        self.assertEqual(results1, results2)

    @staticmethod
    def _evaluate_model(arguments):
        model_config_path = getattr(arguments, 'model_config_path', None)
        if model_config_path is None:
            model_config = {}
        else:
            with open(model_config_path) as f:
                model_config = json.load(f)
        model = get_model(arguments.model, model_config, seed=arguments.model_seed)

        train_data, test_data = get_train_and_test_data(
            arguments.train_path, arguments.test_path, arguments.test_size, arguments.split_seed,
            arguments.metafeature_subset, arguments.cache_dir, arguments.no_cache
        )
        results = []
        for problem_name in getattr(arguments, 'problem'):
            problem = get_problem(problem_name, **vars(arguments))
            results.append(evaluate(
                problem, model, model_config, train_data, test_data
            ))
        return results
