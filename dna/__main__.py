import argparse
import collections
import hashlib
import json
import os
import random
import sys
import tarfile
import traceback
import typing
import uuid
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm
from tuningdeap import TuningDeap

from dna import plot, utils
from dna.data import get_data, preprocess_data, split_data_by_group, group_json_objects
from dna import metrics
from dna.models import get_model, get_model_class
from dna.models.base_models import ModelBase
from dna.problems import get_problem, ProblemBase


MODEL_ID_ORDER = (
    'mean_regression', 'random', 'linear_regression', 'autosklearn', 'random_forest', 'meta_autosklearn', 'lstm',
    'attention_regression', 'daglstm_regression', 'dag_attention_regression', 'dna_regression'
)
MODEL_NAME_ORDER = tuple(get_model_class(model_id).name for model_id in MODEL_ID_ORDER)


def configure_split_parser(parser):
    parser.add_argument(
        '--data-path', type=str, action='store', required=True,
        help='path of data to split'
    )
    parser.add_argument(
        '--train-path', type=str, action='store', default=None,
        help='path to write train data'
    )
    parser.add_argument(
        '--test-path', type=str, action='store', default=None,
        help='path to write test data'
    )
    parser.add_argument(
        '--test-size', type=int, action='store', default=1,
        help='the number of datasets in the test split'
    )
    parser.add_argument(
        '--split-seed', type=int, action='store', default=0,
        help='seed used to split the data into train and test sets'
    )


def split_handler(arguments: argparse.Namespace):
    data_path = getattr(arguments, 'data_path')
    data = get_data(data_path)
    train_data, test_data = split_data_by_group(
        data, 'dataset_id', 'pipeline.steps.name', arguments.test_size, arguments.split_seed
    )

    train_path = getattr(arguments, 'train_path')
    if train_path is None:
        dirname, data_filename = data_path.rsplit(os.path.sep, 1)
        data_filename, ext = data_filename.split('.', 1)
        train_path = os.path.join(dirname, data_filename + '_train.json')

    test_path = getattr(arguments, 'test_path')
    if test_path is None:
        dirname, data_filename = data_path.rsplit(os.path.sep, 1)
        data_filename, ext = data_filename.split('.', 1)
        test_path = os.path.join(dirname, data_filename + '_test.json')

    with open(train_path, 'w') as f:
        json.dump(train_data, f, separators=(',',':'))

    with open(test_path, 'w') as f:
        json.dump(test_data, f, separators=(',',':'))


def configure_evaluate_parser(parser):
    parser.add_argument(
        '--train-path', type=str, action='store', required=True,
        help='path to read the train data'
    )
    parser.add_argument(
        '--test-path', type=str, action='store', default=None,
        help='path to read the test data; if not provided, train data will be split'
    )
    parser.add_argument(
        '--test-size', type=int, action='store', default=1,
        help='the number of datasets in the test split'
    )
    parser.add_argument(
        '--split-seed', type=int, action='store', default=0,
        help='seed used to split the data into train and test sets'
    )
    parser.add_argument(
        '--problem', nargs='+', required=True,
        choices=['regression', 'rank'],
        help='the type of problem'
    )
    parser.add_argument(
        '--model', type=str, action='store', required=True,
        help='the python path to the model class'
    )
    parser.add_argument(
        '--model-config-path', type=str, default=None,
        help='path to a json file containing the model configuration values'
    )
    parser.add_argument(
        '--model-seed', type=int, help='seed used to control the random state of the model'
    )
    parser.add_argument(
        '--verbose', default=False, action='store_true'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='directory path to write outputs for this model run'
    )
    parser.add_argument(
        '--cache-dir', type=str, default='.cache',
        help='directory path to write outputs for this model run'
    )
    parser.add_argument(
        '--no-cache', default=False, action='store_true',
        help='when set, do not use cached preprocessed data'
    )
    parser.add_argument(
        '--metafeature-subset', type=str, default='all', choices=['all', 'landmarkers', 'non-landmarkers']
    )
    parser.add_argument(
        '--use-ootsp', default=False, action='store_true',
        help='when set, enables out-of-training-set pielines (ootsp) mode. discard some pipelines from the training' +\
            ' data and evaluate the model twice: once with test data that contains only in-training-set pipelines ' +\
            'and once with only out-of-training-set pipelines'
    )
    parser.add_argument(
        '--ootsp-split-ratio', type=float, default=0.5,
        help='Used with --use-ootsp to set the ratio of pipelines that will be in the training set'
    )
    parser.add_argument(
        '--ootsp-split-seed', type=int, action='store', default=2,
        help='Seed used with --use-ootsp used to split the train and test sets into ootsp sets'
    )
    parser.add_argument(
        '--skip-test-ootsp', default=False, action='store_true',
        help='Used with --use-ootsp evaluate the model using the ootsp splits, but only the test in-training-set ' +\
            'pipelines. This is useful to compare models that cannot make predictions on ootsp'
    )
    parser.add_argument(
        '--skip-test', default=False, action='store_true', help='skip evaluation of the model on the test data'
    )


class EvaluateResult:

    def __init__(
        self, train_predictions, fit_time, train_predict_time, train_scores, test_predictions, test_predict_time,
        test_scores, ootsp_test_predictions, ootsp_test_predict_time, ootsp_test_scores, problem_name, group_scores_key
    ):
        self.problem_name = problem_name
        self.group_scores_key = group_scores_key
        self.train_predictions = train_predictions
        self.fit_time = fit_time
        self.train_predict_time = train_predict_time
        self.train_scores = train_scores
        self.test_predictions = test_predictions
        self.test_predict_time = test_predict_time
        self.test_scores = test_scores
        self.ootsp_test_predictions = ootsp_test_predictions
        self.ootsp_test_predict_time = ootsp_test_predict_time
        self.ootsp_test_scores = ootsp_test_scores

    def __str__(self):
        results_shallow_copy = self.__dict__
        results = {
            'fit_time': results_shallow_copy['fit_time'],
            'train_predict_time': results_shallow_copy['train_predict_time']
        }

        def get_aggregate_scores(results_copy: dict, problem_name: str, phase: str):
            aggregate_scores_copy = results_copy[phase]['aggregate_scores']
            if problem_name == 'regression':
                return aggregate_scores_copy
            elif problem_name == 'rank':
                aggregate_scores = {
                    'spearman_correlation_mean': aggregate_scores_copy['spearman_correlation_mean'],
                    'spearman_correlation_std_dev': aggregate_scores_copy['spearman_correlation_std_dev'],
                    'spearman_p_value_mean': aggregate_scores_copy['spearman_p_value_mean'],
                    'spearman_p_value_std_dev': aggregate_scores_copy['spearman_p_value_std_dev']
                }
                return aggregate_scores

        train_scores = {
            'total_scores': results_shallow_copy['train_scores']['total_scores'],
            'aggregate_scores': get_aggregate_scores(results_shallow_copy, self.problem_name, 'train_scores')
        }
        results['train_scores'] = train_scores

        results['test_predict_time'] = results_shallow_copy['test_predict_time']
        test_scores = {
            'total_scores': results_shallow_copy['test_scores']['total_scores'],
            'aggregate_scores': get_aggregate_scores(results_shallow_copy, self.problem_name, 'test_scores')
        }
        results['test_scores'] = test_scores

        return json.dumps(results, indent=4)

    def _to_json_for_eq(self):
        return {
            'train_predictions': self.train_predictions,
            'train_scores': self.train_scores,
            'test_predictions': self.test_predictions,
            'test_scores': self.test_scores,
            'ootsp_test_predictions': self.ootsp_test_predictions,
            'ootsp_test_scores': self.ootsp_test_scores,
        }

    def __eq__(self, other):
        self_json = self._to_json_for_eq()
        other_json = other._to_json_for_eq()
        return json.dumps(self_json, sort_keys=True) == json.dumps(other_json, sort_keys=True)


def evaluate(
    problem: ProblemBase, model: ModelBase, model_config: typing.Dict, train_data: typing.Dict, test_data: typing.Dict,
    ootsp_test_data: typing.Dict = None, *, verbose: bool = False, model_output_dir: str = None, plot_dir: str = None
):
    train_predictions, fit_time, train_predict_time = problem.fit_predict(
        train_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
    )
    train_scores = problem.score(train_predictions, train_data)

    if plot_dir is not None:
        problem.plot(train_predictions, train_data, train_scores, os.path.join(plot_dir, 'train'))

    test_predictions = None
    test_predict_time = None
    test_scores = None
    if test_data is not None:
        test_predictions, test_predict_time = problem.predict(
            test_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
        )
        test_scores = problem.score(test_predictions, test_data)

        if plot_dir is not None:
            problem.plot(test_predictions, test_data, test_scores, os.path.join(plot_dir, 'test'))

    ootsp_test_predictions = None
    ootsp_test_predict_time = None
    ootsp_test_scores = None
    if ootsp_test_data is not None:
        ootsp_test_predictions, ootsp_test_predict_time = problem.predict(
            ootsp_test_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
        )
        ootsp_test_scores = problem.score(ootsp_test_predictions, ootsp_test_data)

        if plot_dir is not None:
            problem.plot(
                ootsp_test_predictions, ootsp_test_data, ootsp_test_scores, os.path.join(plot_dir, 'ootsp_test')
            )

    return EvaluateResult(
        train_predictions, fit_time, train_predict_time, train_scores, test_predictions, test_predict_time,
        test_scores, ootsp_test_predictions, ootsp_test_predict_time, ootsp_test_scores, problem.problem_name,
        problem.group_scores_key
    )


def handle_evaluate(model_config: typing.Dict, arguments: argparse.Namespace):
    run_id = str(uuid.uuid4())
    git_commit = utils.get_git_commit_hash()

    output_dir = arguments.output_dir
    model_output_dir = None
    plot_dir = None
    if output_dir is not None:
        output_dir = os.path.join(getattr(arguments, 'output_dir'), run_id)
        model_output_dir = os.path.join(output_dir, 'model')
        os.makedirs(model_output_dir)
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir)

        record_run(run_id, git_commit, output_dir, arguments=arguments, model_config=model_config)

    train_data, test_data = get_train_and_test_data(
        arguments.train_path, arguments.test_path, arguments.test_size, arguments.split_seed,
        arguments.metafeature_subset, arguments.cache_dir, arguments.no_cache
    )
    if arguments.skip_test:
        test_data = None
    ootsp_test_data = None
    if arguments.use_ootsp:
        train_data, test_data, ootsp_test_data = get_ootsp_split_data(
            train_data, test_data, arguments.ootsp_split_ratio, arguments.ootsp_split_seed
        )
        if arguments.skip_test_ootsp:
            ootsp_test_data = None

    model_id = getattr(arguments, 'model')
    if arguments.model_seed is None:
        arguments.model_seed = random.randint(0, 2**32-1)
    model = get_model(model_id, model_config, seed=arguments.model_seed)

    result_scores = []
    for problem_name in getattr(arguments, 'problem'):
        problem = get_problem(problem_name, **vars(arguments))
        if problem.model_is_supported(model):
            if arguments.verbose:
                print('{} {} {}'.format(model_id, problem_name, run_id))

            evaluate_result = evaluate(
                problem, model, model_config, train_data, test_data, ootsp_test_data, verbose=arguments.verbose,
                model_output_dir=model_output_dir, plot_dir=plot_dir
            )
            result_scores.append({
                'problem_name': problem_name,
                'model_id': model_id,
                **evaluate_result.__dict__,
            })

            if arguments.verbose:
                print(evaluate_result, '\n')

    if output_dir is not None:
        record_run(run_id, git_commit, output_dir, arguments=arguments, model_config=model_config, scores=result_scores)

    if output_dir is not None:
        if not os.listdir(model_output_dir):
            os.rmdir(model_output_dir)
        if not os.listdir(plot_dir):
            os.rmdir(plot_dir)

    return result_scores


def get_model_config(model_config_path):
    if model_config_path is None or not os.path.isfile(model_config_path):
        model_config = {}
    else:
        with open(model_config_path) as f:
            model_config = json.load(f)
    return model_config


def evaluate_handler(arguments: argparse.Namespace):
    model_config = get_model_config(getattr(arguments, 'model_config_path', None))
    handle_evaluate(model_config, arguments)


def configure_tuning_parser(parser):
    parser.add_argument(
        '--tuning-config-path', type=str, action='store', required=True,
        help='the directory to read in the tuning config'
    )
    parser.add_argument(
        '--objective', type=str, action='store', required=True,
        choices=['rmse', 'pearson', 'spearman', 'ndcg', 'ndcg_at_k', 'regret', 'regret_at_k']
    )
    parser.add_argument(
        '--k', type=int, action='store', help='the number of pipelines to rank'
    )
    parser.add_argument(
        '--tuning-output-dir', type=str, default=None,
        help='directory path to write outputs from tuningDEAP'
    )
    parser.add_argument(
        '--n-generations', type=int, default=1,
        help='How many generations to tune for'
    )
    parser.add_argument(
        '--population-size', type=int, default=1,
        help='the number of individuals to generate each population'
    )
    configure_evaluate_parser(parser)


def _get_tuning_objective(arguments: argparse.Namespace):
    """Creates the objective function used for tuning.

    Returns
    -------
    (objective, minimize): tuple(Callable, bool)
        a tuple containing the objective function and a Boolean indicating whether the objective should be minimized
    """
    if arguments.objective == 'rmse':
        score_problem = 'regression'
        score_path = ('test_scores', 'total_scores', 'rmse')
        minimize = True
    elif arguments.objective == 'pearson':
        score_problem = 'regression'
        score_path = ('test_scores', 'total_scores', 'pearson_correlation')
        minimize = False
    elif arguments.objective == 'spearman':
        score_problem = 'rank'
        score_path = ('test_scores', 'aggregate_scores', 'spearman_correlation_mean')
        minimize = False
    elif arguments.objective == 'ndcg':
        score_problem = 'rank'
        score_path = ('test_scores', 'total_scores', 'ndcg_at_k_mean')
        minimize = False
    elif arguments.objective == 'ndcg_at_k':
        score_problem = 'rank'
        score_path = ('test_scores', 'aggregate_scores', 'ndcg_at_k_mean', arguments.k)
        minimize = False
    elif arguments.objective == 'regret':
        score_problem = 'rank'
        score_path = ('test_scores', 'total_scores', 'regret_at_k_mean')
        minimize = True
    elif arguments.objective == 'regret_at_k':
        score_problem = 'rank'
        score_path = ('test_scores', 'aggregate_scores', 'regret_at_k_mean', arguments.k)
        minimize = True
    else:
        raise ValueError('unknown objective {}'.format(arguments.objective))

    def objective(model_config):
        try:
            result_scores = handle_evaluate(model_config, arguments)
            for scores in result_scores:
                if scores['problem_name'] == score_problem:
                    score = scores
                    for key in score_path:
                        score = score[key]
                    return (score,)
            raise ValueError('{} problem required for "{}" objective'.format(score_problem, arguments.objective))
        except Exception as e:
            traceback.print_exc()
            raise e from e

    return objective, minimize


def tuning_handler(arguments: argparse.Namespace):
    # gather config files
    model_config = get_model_config(getattr(arguments, 'model_config_path', None))
    with open(arguments.tuning_config_path, 'r') as file:
        tuning_config = json.load(file)

    objective, minimize = _get_tuning_objective(arguments)
    tuning_run_id = str(uuid.uuid4())
    tuning_output_dir = os.path.join(arguments.tuning_output_dir, arguments.model, tuning_run_id)
    if not os.path.isdir(tuning_output_dir):
        os.makedirs(tuning_output_dir)

    tune = TuningDeap(
        objective, tuning_config, model_config, minimize=minimize, output_dir=tuning_output_dir,
        verbose=arguments.verbose, population_size=arguments.population_size, n_generations=arguments.n_generations
    )
    best_config, best_score = tune.run_evolutionary()
    if arguments.verbose:
        print('The best config found was {} with a score of {}'.format(
            ' '.join([str(item) for item in best_config]), best_score
        ))


def configure_rescore_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--results-path', type=str, action='store', help='path of file to rescore'
    )
    parser.add_argument(
        '--results-dir', type=str, help='directory of results to rescore'
    )
    parser.add_argument(
        '--result-paths-csv', type=str, help='path to csv containing paths of files to rescore'
    )
    parser.add_argument(
        '--output-dir', type=str, action='store', default='./rescore',
        help='the base directory to write the recomputed scores and plots'
    )
    parser.add_argument(
        '--plot', default=False, action='store_true', help='whether to remake the plots'
    )


def rescore_handler(arguments: argparse.Namespace):
    if arguments.results_path is not None:
        result_paths = [arguments.results_path]
    elif arguments.results_dir is not None:
        result_paths = get_result_paths_from_dir(arguments.results_dir)
    elif arguments.result_paths_csv is not None:
        result_paths = get_result_paths_from_csv(arguments.result_paths_csv)

    for results_path in tqdm(sorted(result_paths)):
        handle_rescore(results_path, arguments.output_dir, arguments.plot)


def handle_rescore(results_path: str, output_dir: str, plot: bool):
    with open(results_path) as f:
        results = json.load(f)

    if 'scores' not in results:
        print('no scores for {}'.format(results_path))
        return

    train_data, test_data = get_train_and_test_data(
        results['arguments']['train_path'], results['arguments']['test_path'], results['arguments']['test_size'],
        results['arguments']['split_seed'], results['arguments']['metafeature_subset'],
        results['arguments']['cache_dir'], results['arguments']['no_cache']
    )

    # Create the output directory for the results of the re-score
    output_dir = os.path.join(output_dir, results['id'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if plot:
        plot_dir = os.path.join(output_dir, 'plots')
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)

    # For each problem, re-score using the predictions and data and ensure the scores are the same as before
    for problem_scores in results.get('scores', []):
        problem = get_problem(problem_scores['problem_name'], **results['arguments'])
        if problem is None:
            continue

        train_predictions = problem_scores['train_predictions']
        train_rescores = problem.score(train_predictions, train_data)
        problem_scores['train_scores'] = train_rescores
        if plot:
            problem.plot(train_predictions, train_data, train_rescores, os.path.join(plot_dir, 'train'))

        test_predictions = problem_scores['test_predictions']
        test_rescores = problem.score(test_predictions, test_data)
        problem_scores['test_scores'] = test_rescores
        if plot:
            problem.plot(test_predictions, test_data, test_rescores, os.path.join(plot_dir, 'test'))

    # Save the re-scored json file
    rescore_path = os.path.join(output_dir, 'run.json')
    with open(rescore_path, 'w') as f:
        json.dump(results, f, indent=4, sort_keys=True)


def get_train_and_test_data(
    train_path, test_path, test_size, split_seed, metafeature_subset, cache_dir, no_cache: bool
):
    data_arg_str = str(train_path) + str(test_path) + str(test_size) + str(split_seed) + str(metafeature_subset)
    cache_id = hashlib.sha256(data_arg_str.encode('utf8')).hexdigest()
    cache_dir = os.path.join(cache_dir, cache_id)
    train_cache_path = os.path.join(cache_dir, 'train.json')
    test_cache_path = os.path.join(cache_dir, 'test.json')

    load_cached_data = (not no_cache) and (os.path.isdir(cache_dir))

    # determine whether to load raw or cached data
    if load_cached_data:
        in_train_path = train_cache_path
        in_test_path = test_cache_path
    else:
        in_train_path = train_path
        in_test_path = test_path

    # when loading raw data and test_path is not provided, split train into train and test data
    train_data = get_data(in_train_path)
    if in_test_path is None:
        assert not load_cached_data
        train_data, test_data = split_data_by_group(
            train_data, 'dataset_id', 'pipeline.steps.name', test_size, split_seed
        )
    else:
        test_data = get_data(in_test_path)

    if not load_cached_data:
        train_data, test_data = preprocess_data(train_data, test_data, metafeature_subset)
        if not no_cache:
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir)
            with open(train_cache_path, 'w') as f:
                json.dump(train_data, f, separators=(',',':'))
            with open(test_cache_path, 'w') as f:
                json.dump(test_data, f, separators=(',',':'))

    return train_data, test_data


def get_ootsp_split_data(train_data, test_data, split_ratio, split_seed):
    # TODO: Why is `split_ratio` necessary here? It only seems to be
    # reducing the size of the training data set. Also, when `split_ratio < 1`,
    # we can no longer ensure that all the primitives are present in the training set.
    train_pipeline_ids = sorted(set(instance['pipeline_id'] for instance in train_data))
    k = int(split_ratio * len(train_pipeline_ids))

    rnd = random.Random()
    rnd.seed(split_seed)
    in_train_set_pipeline_ids = set(rnd.choices(train_pipeline_ids, k=k))

    new_train_data = [instance for instance in train_data if instance['pipeline_id'] in in_train_set_pipeline_ids]
    itsp_test_data = []
    ootsp_test_data = []
    for instance in test_data:
        if instance['pipeline_id'] in in_train_set_pipeline_ids:
            itsp_test_data.append(instance)
        else:
            ootsp_test_data.append(instance)

    return new_train_data, itsp_test_data, ootsp_test_data


def record_run(
    run_id: str, git_commit: str, output_dir: str, *, arguments: argparse.Namespace, model_config: typing.Dict,
    scores: typing.Dict = None
):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, 'run.json')

    run = {
        'id': run_id,
        'git_commit': git_commit,
        'arguments': arguments.__dict__,
        'model_config': model_config,
    }
    if scores is not None:
        run['scores'] = scores

    with open(path, 'w') as f:
        json.dump(run, f, indent=4, sort_keys=True, cls=utils.NumpyJSONEncoder)


def configure_report_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--results-dir', type=str, help='directory containing results'
    )
    parser.add_argument(
        '--result-paths-csv', type=str, help='path to csv containing paths to results'
    )
    parser.add_argument(
        '--report-dir', type=str, default='./report', help='directory to output score reports'
    )


def get_regression_report_path(report_dir: str):
    return os.path.join(report_dir, 'regression_leaderboard.csv')


def get_rank_report_path(report_dir: str):
    return os.path.join(report_dir, 'rank_leaderboard.csv')


def report_handler(arguments: argparse.Namespace):
    if arguments.results_dir is not None:
        result_paths = get_result_paths_from_dir(arguments.results_dir)
    elif arguments.result_paths_csv is not None:
        result_paths = get_result_paths_from_csv(arguments.result_paths_csv)
    else:
        raise ValueError('one of --results-dir or result-paths-csv must be provided')

    regression_results, rank_results = load_results(result_paths)
    regression_leaderboard = make_leaderboard(regression_results, 'test.total_scores.rmse_mean_over_runs', min)
    rank_leaderboard = make_leaderboard(rank_results, 'test.total_scores.ndcg_at_k_mean_mean_over_runs', max)

    if not os.path.isdir(arguments.report_dir):
        os.makedirs(arguments.report_dir)

    rank_scores_by_metric_by_model, rank_model_colors = _get_score_distributions_by_metric_by_model(rank_results)
    regression_scores_by_metric_by_model, regression_model_colors = _get_score_distributions_by_metric_by_model(regression_results)

    plot_ndcg_over_k(rank_leaderboard, arguments.report_dir)
    plot_regret_over_k(rank_leaderboard, arguments.report_dir)
    plot_n_correct_over_k(rank_leaderboard, arguments.report_dir)

    plot_spearman_distribution(rank_scores_by_metric_by_model['spearman_correlation'], rank_model_colors, arguments.report_dir)
    plot_spearman_p_values_distribution(rank_scores_by_metric_by_model['spearman_p_value'], rank_model_colors, arguments.report_dir)
    plot_pearson_distribution(regression_scores_by_metric_by_model['pearson_correlation'], regression_model_colors, arguments.report_dir)
    plot_pearson_p_values_distribution(regression_scores_by_metric_by_model['pearson_p_value'], regression_model_colors, arguments.report_dir)
    plot_rmse_distribution(regression_scores_by_metric_by_model['rmse'], regression_model_colors, arguments.report_dir)

    for k in [1, 25, 100, -1]:
        plot_ndcg_distribution(rank_scores_by_metric_by_model['ndcg_at_k'], k, rank_model_colors, arguments.report_dir)
        plot_regret_distribution(rank_scores_by_metric_by_model['regret_at_k'], k, rank_model_colors, arguments.report_dir)
        plot_n_correct_distribution(rank_scores_by_metric_by_model['n_correct_at_k'], k, rank_model_colors, arguments.report_dir)
        for col_name in rank_leaderboard.columns:
            if 'aggregate_scores' in col_name and 'at_k' in col_name:
                new_col_name = col_name.replace('at_k', 'at_{}'.format(k))
                rank_leaderboard[new_col_name] = np.nan
                for model_name in rank_leaderboard['model_name']:
                    scores_by_k = rank_leaderboard[col_name][rank_leaderboard['model_name'] == model_name].iloc[0]
                    score_at_k = scores_by_k[k]
                    rank_leaderboard[new_col_name][rank_leaderboard['model_name'] == model_name] = score_at_k

    rank_columns = list(rank_leaderboard.columns)
    for col_name in rank_leaderboard.columns:
        if 'aggregate_scores' in col_name and 'at_k' in col_name:
            rank_columns.remove(col_name)
        elif 'train' in col_name:
            rank_columns.remove(col_name)
        elif 'scores_by_dataset_id' in col_name:
            rank_columns.remove(col_name)
    rank_leaderboard = rank_leaderboard[rank_columns]

    regression_columns = list(regression_leaderboard.columns)
    for col_name in regression_leaderboard.columns:
        if 'train' in col_name:
            regression_columns.remove(col_name)
        elif 'scores_by_dataset_id' in col_name:
            regression_columns.remove(col_name)
        elif 'aggregate_scores' in col_name:
            regression_columns.remove(col_name)
    regression_leaderboard = regression_leaderboard[regression_columns]

    path = get_regression_report_path(arguments.report_dir)
    regression_leaderboard.to_csv(path, index=False)
    path = get_rank_report_path(arguments.report_dir)
    rank_leaderboard.to_csv(path, index=False)

    save_result_paths_csv(regression_results, rank_results, report_dir=arguments.report_dir)


def get_result_paths_from_dir(results_dir: str):
    return get_result_paths([os.path.join(results_dir, dir_) for dir_ in os.listdir(results_dir)])


def get_result_paths_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, header=None)
    return get_result_paths(df[0])


def get_result_paths(result_dirs: typing.Sequence[str]):
    result_paths = []
    for dir_ in result_dirs:
        path = os.path.join(dir_, 'run.json')
        if os.path.isfile(path + '.tar.gz') and not os.path.isfile(path):
            # unzip if needed
            tar = tarfile.open(path + '.tar.gz', 'r:gz')
            tar.extractall()
            tar.close()
        if os.path.isfile(path):
            result_paths.append(path)
        else:
            print(f'file does not exist: {path}')
    return result_paths


def load_results(result_paths: typing.Sequence[str]) -> pd.DataFrame:
    regression_results = []
    rank_results = []
    for result_path in tqdm(result_paths):
        result = load_result(result_path)
        if result == {}:
            print(f'no results for {result_path}')
        else:
            if 'regression' in result:
                regression_results.append(result['regression'])
            if 'rank' in result:
                rank_results.append(result['rank'])
    return pd.DataFrame(regression_results), pd.DataFrame(rank_results)


def load_result(result_path: str):
    with open(result_path) as f:
        run = json.load(f)

    result = {}
    try:
        for problem_scores in run.get('scores', []):
            problem_name = problem_scores['problem_name']
            if problem_name in ['regression', 'rank']:
                parsed_scores = parse_scores(problem_scores)
                parsed_scores['path'] = os.path.dirname(result_path)
                result[problem_name] = parsed_scores
    except Exception as e:
        traceback.print_exc()
        print('failed to load {}'.format(result_path))
    return result


def parse_scores(scores: typing.Dict):
    return {
        'model_id': scores['model_id'],
        **utils.flatten(scores['train_scores'], 'train'),
        **utils.flatten(scores['test_scores'], 'test')
    }


def make_leaderboard(results: pd.DataFrame, score_col: str, opt: typing.Callable):
    id_col = 'model_id'
    grouped_results = results.groupby(id_col)
    counts = grouped_results.size().astype(int)
    counts.name = 'count'
    leader_indices = grouped_results[score_col].transform(opt) == results[score_col]
    leaderboard = results[leader_indices].drop_duplicates([id_col, score_col])
    leaderboard.sort_values([score_col, id_col], ascending=opt==min, inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)
    leaderboard = leaderboard.join(counts, on=id_col)

    _add_model_name_and_color_to_leaderboard(leaderboard)

    # sort rows
    leaderboard.model_id = leaderboard.model_id.astype('category')
    leaderboard.model_id.cat.set_categories(
        MODEL_ID_ORDER,
        inplace=True
    )
    leaderboard.sort_values(['model_id'], inplace=True)

    # sort columns
    columns = list(leaderboard.columns)
    columns.remove(id_col)
    columns.remove(counts.name)
    columns.remove('model_name')
    columns.remove('model_color')
    leaderboard = leaderboard[['model_name', 'model_color', counts.name] + sorted(columns)]

    return leaderboard.round(8)


def _add_model_name_and_color_to_leaderboard(leaderboard):
    model_ids = list(leaderboard['model_id'])
    model_names = []
    model_colors = []
    for i, model_id in enumerate(model_ids):
        model_class = get_model_class(model_id)
        if hasattr(model_class, 'name') and hasattr(model_class, 'color'):
            model_names.append(model_class.name)
            model_colors.append(model_class.color)
        else:
            leaderboard.drop(leaderboard.index[i], inplace=True)
    leaderboard['model_name'] = model_names
    leaderboard['model_color'] = model_colors


def plot_ndcg_over_k(rank_report: pd.DataFrame, output_dir: str):
    plot_path = os.path.join(output_dir, 'ndcg_over_k.pdf')
    plot.plot_at_k_scores_over_k(
        rank_report['model_name'], rank_report['test.aggregate_scores.ndcg_at_k_mean_mean_over_runs'],
        rank_report['test.aggregate_scores.ndcg_at_k_mean_std_dev_over_runs'],
        rank_report['model_color'], plot_path, 'NDCG@k', None
    )


def plot_regret_over_k(rank_report: pd.DataFrame, output_dir: str):
    plot_path = os.path.join(output_dir, 'regret_over_k.pdf')
    plot.plot_at_k_scores_over_k(
        rank_report['model_name'], rank_report['test.aggregate_scores.regret_at_k_mean_mean_over_runs'],
        rank_report['test.aggregate_scores.regret_at_k_mean_std_dev_over_runs'],
        rank_report['model_color'], plot_path, 'Regret@k', None
    )


def plot_n_correct_over_k(rank_report: pd.DataFrame, output_dir: str):
    plot_path = os.path.join(output_dir, 'topk_over_k.pdf')
    plot.plot_at_k_scores_over_k(
        rank_report['model_name'], rank_report['test.aggregate_scores.n_correct_at_k_mean_mean_over_runs'],
        rank_report['test.aggregate_scores.n_correct_at_k_mean_std_dev_over_runs'],
        rank_report['model_color'], plot_path, 'Top-K@k', None
    )


def _get_score_distributions_by_metric_by_model(results: pd.DataFrame):
    scores_by_metric_by_model = {}

    def insert_score(path: typing.Sequence, score_value):
        if np.isnan(score_value).any():
            return

        nonlocal scores_by_metric_by_model

        obj = scores_by_metric_by_model
        for i, key in enumerate(path):
            if key not in obj:
                if i < len(path) - 1:
                    obj[key] = {}
                else:
                    obj[key] = []
            obj = obj[key]

        if type(score_value) == list:
            obj.extend(score_value)
        else:
            obj.append(score_value)

    model_colors = {}
    for row_number, row in results.iterrows():
        model_class = get_model_class(row['model_id'])
        model_name = model_class.name
        model_colors[model_name] = model_class.color
        for column_name, cell_value in row.iteritems():
            if column_name.startswith('test.scores_by_dataset_id'):
                assert column_name.endswith('_by_run')

                _, _, dataset_id, raw_metric_name = column_name.split('.')
                metric_name = raw_metric_name.replace('_by_run', '')

                if 'at_k' in metric_name:
                    assert type(cell_value) == list and type(cell_value[0]) == list
                    for run_scores_at_k in cell_value:
                        for i, score_value in enumerate(run_scores_at_k):
                            k = i+1  # index 0 contains values for k=1
                            insert_score((metric_name, k, model_name), score_value)
                            if k == len(run_scores_at_k):
                                assert run_scores_at_k[i] == run_scores_at_k[-1]
                                insert_score((metric_name, -1, model_name), score_value)
                else:
                    insert_score((metric_name, model_name), cell_value)

    return scores_by_metric_by_model, model_colors


def _plot_scores_at_k_distribution(score_name: str, scores_at_k_by_model: pd.DataFrame, k: typing.Sequence[int], model_colors, output_dir: str):
    scores_at_k_by_model = scores_at_k_by_model[k]
    ylabel = score_name
    if k > 0:
        ylabel += '@{}'.format(k)
    title = 'Distributions of {}'.format(ylabel)
    plot_path = os.path.join(output_dir, '{}_at_{}_distributions.pdf'.format(score_name.lower(), k))
    plot.plot_violin_of_score_distributions(scores_at_k_by_model, model_colors, MODEL_NAME_ORDER, ylabel, title, plot_path)


def plot_ndcg_distribution(scores_at_k_by_model: pd.DataFrame, k: typing.Sequence[int], model_colors, output_dir: str):
    _plot_scores_at_k_distribution('NDCG', scores_at_k_by_model, k, model_colors, output_dir)


def plot_regret_distribution(scores_at_k_by_model: pd.DataFrame, k: typing.Sequence[int], model_colors, output_dir: str):
    _plot_scores_at_k_distribution('Regret', scores_at_k_by_model, k, model_colors, output_dir)


def plot_n_correct_distribution(scores_at_k_by_model: pd.DataFrame, k: typing.Sequence[int], model_colors, output_dir: str):
    _plot_scores_at_k_distribution('Top-K', scores_at_k_by_model, k, model_colors, output_dir)


def _plot_scores_distribution(score_name, scores_by_model, model_colors, output_dir):
    ylabel = score_name
    title = 'Distributions of {}'.format(ylabel)
    plot_path = os.path.join(output_dir, '{}_distributions.pdf'.format(score_name.lower()))
    plot.plot_violin_of_score_distributions(scores_by_model, model_colors, MODEL_NAME_ORDER, ylabel, title, plot_path)


def plot_spearman_distribution(scores_by_model, model_colors, output_dir: str):
    _plot_scores_distribution('Spearman Correlation', scores_by_model, model_colors, output_dir)


def plot_spearman_p_values_distribution(scores_by_model, model_colors, output_dir: str):
    _plot_scores_distribution('Spearman Correlation p-values', scores_by_model, model_colors, output_dir)


def plot_pearson_distribution(scores_by_model, model_colors, output_dir: str):
    _plot_scores_distribution('Pearson Correlation', scores_by_model, model_colors, output_dir)


def plot_pearson_p_values_distribution(scores_by_model, model_colors, output_dir: str):
    _plot_scores_distribution('Pearson Correlation p-values', scores_by_model, model_colors, output_dir)


def plot_rmse_distribution(scores_by_model, model_colors, output_dir: str):
    _plot_scores_distribution('RMSE', scores_by_model, model_colors, output_dir)


def save_result_paths_csv(*args: pd.DataFrame, report_dir):
    result_paths = []
    for results in args:
        result_paths += results['path'].tolist()
    df = pd.DataFrame(set(result_paths))
    path = os.path.join(report_dir, 'result_paths.csv')
    df.to_csv(path, header=False, index=False)


def configure_agg_results_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--results-dir', type=str, help='directory of results to aggregate'
    )
    parser.add_argument(
        '--result-paths-csv', type=str, help='path to csv containing paths of files to aggregate'
    )
    parser.add_argument(
        '--output-dir', type=str, action='store', default='./agg_results',
        help='the base directory to write the aggregated scores and plots'
    )


def aggregate_result_scores(results_to_agg: typing.List[typing.Dict]):
    # group results by problem and flatten
    problem_name_to_scores_to_agg_map = {}  # problem_name: scores_to_agg
    for result in results_to_agg:
        for problem_scores in result.get('scores', []):
            problem_scores['run_id'] = result['id']
            problem_name = problem_scores['problem_name']
            if problem_name not in problem_name_to_scores_to_agg_map:
                problem_name_to_scores_to_agg_map[problem_name] = []
            problem_name_to_scores_to_agg_map[problem_name].append(utils.flatten(problem_scores))

    agg_scores = []
    for problem_name, problem_scores_to_agg in problem_name_to_scores_to_agg_map.items():
        problem_scores_to_agg_df = pd.DataFrame(problem_scores_to_agg).dropna(axis=1, how='all')
        # record metadata
        flat_agg_problem_scores = {
            'problem_name': problem_name,
            'model_id': problem_scores_to_agg_df['model_id'].iloc[0],
            'aggregated_run_ids': problem_scores_to_agg_df['run_id'].tolist(),
        }
        for col_name, column in problem_scores_to_agg_df.iteritems():
            column_values = column.values.tolist()  # each row is a different run

            # keep scores over all datasets and runs
            if 'scores_by_dataset_id' in col_name:
                flat_agg_problem_scores[col_name+'_by_run'] = column_values

            # compute mean and std dev over all runs for all other scores
            elif 'train_scores' in col_name or 'test_scores' in col_name:
                flat_agg_problem_scores[col_name+'_mean_over_runs'] = np.mean(column_values, axis=0).tolist()
                flat_agg_problem_scores[col_name+'_std_dev_over_runs'] = np.std(column_values, axis=0).tolist()

        agg_scores.append(utils.inflate(flat_agg_problem_scores))

    return agg_scores


def agg_results_handler(arguments: argparse.Namespace):
    if arguments.results_dir is not None:
        result_paths = get_result_paths_from_dir(arguments.results_dir)
    elif arguments.result_paths_csv is not None:
        result_paths = get_result_paths_from_csv(arguments.result_paths_csv)
    else:
        raise ValueError('no results to aggregate')

    results_to_agg = [json.load(open(results_path)) for results_path in tqdm(sorted(result_paths))]
    for i in range(1, len(results_to_agg)):
        model_config_0 = results_to_agg[0]['model_config']
        model_config_1 = results_to_agg[1]['model_config']
        if model_config_0 != model_config_1:
            raise ValueError('Results from different models should not be aggregated')

    results_ids = sorted([result['id'] for result in results_to_agg])
    uuid_namespace = uuid.UUID('a8ec5977-701e-443b-89e7-8974e0e4ea6c')
    run_id = str(uuid.uuid5(uuid_namespace, json.dumps(results_ids)))

    git_commit = utils.get_git_commit_hash()

    output_dir = os.path.join(getattr(arguments, 'output_dir'), run_id)

    agg_scores = aggregate_result_scores(results_to_agg)

    record_run(
        run_id, git_commit, output_dir, arguments=arguments, model_config=results_to_agg[0]['model_config'],
        scores=agg_scores
    )

    result_arguments = results_to_agg[0]['arguments']

    train_data, test_data = get_train_and_test_data(
        result_arguments['train_path'], result_arguments['test_path'], result_arguments['test_size'], result_arguments['split_seed'],
        result_arguments['metafeature_subset'], result_arguments['cache_dir'], result_arguments['no_cache']
    )

    predictions_description = pd.DataFrame(test_data)[['dataset_id', 'pipeline_id', 'test_f1_macro']]
    has_regression_results = False
    for i, results in enumerate(results_to_agg):
        for scores in results.get('scores', []):
            if scores['problem_name'] == 'regression':
                test_predictions = scores['test_predictions']
                if metrics.rmse(test_predictions, predictions_description['test_f1_macro']) != scores['test_scores']['total_scores']['rmse']:
                    raise ValueError('rmse is not deterministic based on results and results arguments')

                predictions_description['predictions{}'.format(i)] = test_predictions

                has_regression_results = True

    if has_regression_results:
        predictions_path = os.path.join(output_dir, 'predictions.csv')
        predictions_description.to_csv(predictions_path, index=False)


def handler(arguments: argparse.Namespace, parser: argparse.ArgumentParser):
    if arguments.command == 'split-data':
        split_handler(arguments)

    elif arguments.command == 'evaluate':
        evaluate_handler(arguments)

    elif arguments.command == 'rescore':
        rescore_handler(arguments)

    elif arguments.command == 'tune':
        tuning_handler(arguments)

    elif arguments.command == 'report':
        report_handler(arguments)

    elif arguments.command == 'agg-results':
        agg_results_handler(arguments)

    else:
        raise ValueError('Unknown command: {}'.format(arguments.command))


def main(argv: typing.Sequence):

    parser = argparse.ArgumentParser(prog='dna')

    subparsers = parser.add_subparsers(dest='command', title='command')
    subparsers.required = True

    split_parser = subparsers.add_parser(
        'split-data', help='creates train and test splits of the data'
    )
    configure_split_parser(split_parser)

    evaluate_parser = subparsers.add_parser(
        'evaluate', help='train, score, and save a model'
    )
    configure_evaluate_parser(evaluate_parser)

    rescore_parser = subparsers.add_parser(
        'rescore', help='recompute scores from the file output by evaluate and remake the plots'
    )
    configure_rescore_parser(rescore_parser)

    tuning_parser = subparsers.add_parser(
        'tune', help='recompute scores from the file output by evaluate and remake the plots'
    )
    configure_tuning_parser(tuning_parser)

    report_parser = subparsers.add_parser(
        'report', help='generate a report of the best models'
    )
    configure_report_parser(report_parser)

    agg_results_parser = subparsers.add_parser(
        'agg-results', help='aggregate the results of multiple runs, e.g. average of random model'
    )
    configure_agg_results_parser(agg_results_parser)

    arguments = parser.parse_args(argv[1:])

    handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
