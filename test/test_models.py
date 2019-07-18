import unittest

import argparse
import json

from dna.__main__ import configure_evaluate_parser, evaluate, get_train_and_test_data
from dna.data import get_data
from dna.models.models import get_model
from dna.problems import get_problem


class ModelDeterminismTestCase(unittest.TestCase):

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
            '--train-path', './data/small_classification_train.json',
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

        train_data, test_data = get_train_and_test_data(arguments, data_resolver=get_data)
        results = []
        for problem_name in getattr(arguments, 'problem'):
            problem = get_problem(problem_name, arguments)
            results.append(evaluate(
                problem, model, model_config, train_data, test_data
            ))
        return results
