import unittest
import json
import argparse
import os
import shutil

from dna.__main__ import handle_evaluate, configure_rerun_parser, rerun_handler, EvaluateResult
from test.utils import get_evaluate_args, split_data, get_model_config


class RerunTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data_path_train = 'data/small_classification_train.json'
        raw_data_path =  'data/small_classification.tar.xz'
        split_data(data_path_train, raw_data_path)

    def test_rerun(self):
        # Get the arguments for the evaluate command
        model = 'dna_regression'
        model_config_path = 'model_configs/dna_regression_config.json'
        data_train_path = 'data/small_classification_train.json'
        evaluate_args = get_evaluate_args(model, model_config_path, data_train_path)

        # Get the output directory for this test
        output_dir = 'tmp_output'
        evaluate_args.__setattr__('output_dir', output_dir)

        # Remove the output directory in case it wasn't removed from a previous test
        shutil.rmtree(output_dir, ignore_errors=True)

        # Get the results from the command handler and save the results in a run.json
        model_config = get_model_config(model_config_path)
        run_id = 'abc123'
        results1 = handle_evaluate(model_config, evaluate_args, run_id=run_id)

        # Get the path to the run.json that was saved by the evaluate command
        run_path = os.path.join(output_dir, run_id, 'run.json')

        # Get the results from the rerun command
        parser = argparse.ArgumentParser()
        configure_rerun_parser(parser)
        argv = ['--run-path', run_path]
        arguments = parser.parse_args(argv)
        results2 = rerun_handler(arguments)

        # Remove the output directory that was created by the evaluate command
        shutil.rmtree(output_dir, ignore_errors=True)

        # Ensure the results from the evaluate command and the rerun command are equal
        self.get_evaluate_results(results1)
        self.get_evaluate_results(results2)
        self.assertEqual(results1, results2)

    @staticmethod
    def get_evaluate_results(results):
        # Remove the problem name and model name from the results dictionaries in order to create EvaluateResult objects
        # This enables the results to be compared by assertEqual
        for i, problem_results in enumerate(results):
            del problem_results['problem_name']
            del problem_results['model_name']
            evaluate_result = EvaluateResult(**problem_results)
            results[i] = evaluate_result
