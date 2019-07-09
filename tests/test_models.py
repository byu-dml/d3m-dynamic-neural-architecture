import unittest

import argparse
import json

from dna.__main__ import configure_evaluate_parser
from dna.__main__ import get_train_and_test_data
from dna.data import get_data
from dna.models import get_model
from dna.problems import get_problem


class ModelsTestCase(unittest.TestCase):

    def test_dna_regression_determinism(self):
        self._test_determinism(model='dna_regression', model_config_path='./model_configs/dna_regression_config.json')

    def test_dag_rnn_regression_determinism(self):
        self._test_determinism(model='dagrnn_regression',
                               model_config_path='./model_configs/dagrnn_regression_config.json')

    def _test_determinism(self, model: str, model_config_path: str):
        # Set the arguments for this test
        parser = argparse.ArgumentParser()
        configure_evaluate_parser(parser)
        argv = ['--model', model,
                '--model-config-path', model_config_path,
                '--problem', 'regression', 'rank',
                '--k', '5',
                '--scores', 'top-k-count', 'top-1-regret', 'spearman', 'top-k-regret', 'pearson',
                '--train-path', './data/small_classification_train.json',
                '--test-size', '2',
                '--split-seed', '5460650386',
                '--output-dir', './dev_results',
                ]
        arguments = parser.parse_args(argv)

        # Ensure consistency of results after two runs
        results1 = self._get_results(arguments)
        results2 = self._get_results(arguments)
        self.assertEqual(results1, results2)

    @staticmethod
    def _get_results(arguments):
        # Get the data
        (train_data, test_data) = get_train_and_test_data(arguments, data_resolver=get_data)

        results = []

        # Get the model
        with open(arguments.model_config_path) as f:
            model_config = json.load(f)
        model = get_model(arguments.model, model_config, seed=arguments.model_seed)

        # Get results for each problem to ensure that they are consistent
        for problem_name in arguments.problem:
            problem = get_problem(problem_name)
            if problem_name == 'rank':
                k = getattr(arguments, 'k')
                _, _, train_scores, test_scores, _ = problem.run(
                    train_data, test_data, model, k, getattr(arguments, 'scores'), model_config=model_config,
                    re_fit_model=False, verbose=False, output_dir=arguments.output_dir
                )
            else:
                _, _, train_scores, test_scores, _ = problem.run(
                    train_data, test_data, model, model_config=model_config, re_fit_model=False,
                    verbose=False, output_dir=arguments.output_dir
                )
            results.append(train_scores)
            results.append(test_scores)

        return results


