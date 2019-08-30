import unittest
import json
import argparse
import os
import shutil

from dna.__main__ import (
    handle_evaluate, configure_rerun_parser, rerun_handler, EvaluateResult, tuning_handler, configure_tuning_parser,
    configure_evaluate_parser
)
from test.utils import get_evaluate_args, split_data, get_model_config, get_evaluate_argv, parse_args


def init_tests():
    rerun_output_dir = 'tmp_output'
    rerun_id = 'abc123'
    model = 'lstm'
    model_config_path = 'test/model_configs/lstm_config.json'

    # Split the data
    data_path = 'data/small_classification_train.json'
    raw_data_path = 'data/small_classification.tar.xz'
    split_data(data_path, raw_data_path)

    # Delete the output directory from a previous run if it exists to avoid exceptions
    shutil.rmtree(rerun_output_dir, ignore_errors=True)

    # Get initial results for comparison
    model_config = get_model_config(model_config_path)
    evaluate_argv = get_evaluate_argv(model, model_config_path, data_path)
    evaluate_args = parse_args(evaluate_argv, configure_evaluate_parser)
    evaluate_args.__setattr__('output_dir', rerun_output_dir)
    results1 = handle_evaluate(model_config, evaluate_args, run_id=rerun_id)

    return rerun_output_dir, rerun_id, evaluate_argv, results1


RERUN_OUTPUT_DIR, RERUN_ID, EVALUATE_ARGV, RESULTS1 = init_tests()


class RerunTestCase(unittest.TestCase):

    def test_rerun(self):
        # Get the path to the run.json that was saved by the evaluate command
        run_path = os.path.join(RERUN_OUTPUT_DIR, RERUN_ID, 'run.json')

        # Get the results from the rerun command
        argv = ['--run-path', run_path]
        arguments = parse_args(argv, configure_rerun_parser)
        results2 = rerun_handler(arguments)

        # Remove the output directory that was created by the evaluate command
        shutil.rmtree(RERUN_OUTPUT_DIR, ignore_errors=True)

        # Ensure the results from the evaluate command and the rerun command are equal
        results1 = self.get_evaluate_results(RESULTS1)
        results2 = self.get_evaluate_results(results2)
        self.assertEqual(results1, results2)

    @staticmethod
    def get_evaluate_results(results):
        results = results.copy()
        # Remove the problem name and model name from the results dictionaries in order to create EvaluateResult objects
        # This enables the results to be compared by assertEqual
        for i, problem_results in enumerate(results):
            del problem_results['problem_name']
            del problem_results['model_name']
            evaluate_result = EvaluateResult(**problem_results)
            results[i] = evaluate_result
        return results


class TuningTestCase(unittest.TestCase):

    def test_tuning(self):
        # Make the arguments for the tune command
        tuning_output_dir = 'tune_out'
        objective = 'ndcg@k'
        tuning_argv = [
            '--tuning-config-path', 'tuning_configs/lstm_tuning_config.json',
            '--objective', objective,
            '--tuning-output-dir', tuning_output_dir
        ]
        tuning_argv.extend(EVALUATE_ARGV)
        tuning_args = parse_args(tuning_argv, configure_tuning_parser)

        shutil.rmtree(tuning_output_dir, ignore_errors=True)

        # Test the tuning command
        best_config, best_score = tuning_handler(tuning_args)
        self.assertIsNotNone(best_config)
        orig_score = RESULTS1[-1]['test_scores']['ndcg_at_k']['mean']
        self.assertGreaterEqual(best_score, orig_score)
        self.assertTrue(os.path.isdir(tuning_output_dir))

        shutil.rmtree(tuning_output_dir, ignore_errors=True)
