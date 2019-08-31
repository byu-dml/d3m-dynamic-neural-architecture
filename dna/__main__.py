import argparse
import hashlib
import json
import os
import random
import sys
import typing
import uuid
import warnings

import numpy as np
import pandas as pd

from tuningdeap import TuningDeap

from dna import utils
from dna.data import get_data, preprocess_data, split_data_by_group, group_json_objects
from dna.models import get_model
from dna.models.base_models import ModelBase
from dna.problems import get_problem, ProblemBase


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
    train_data, test_data = split_data_by_group(data, 'dataset_id', arguments.test_size, arguments.split_seed)

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
        '--k', type=int, action='store', default=10,
        help='the number of pipelines to rank'
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
        '--model-seed', type=int, default=1,
        help='seed used to control the random state of the model'
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
        test_scores, ootsp_test_predictions, ootsp_test_predict_time, ootsp_test_scores
    ):
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

    def __eq__(self, other):
        result = self.train_predictions == other.train_predictions
        result &= self.train_scores == other.train_scores
        result &= self.test_predictions == other.test_predictions
        result &= self.test_scores == other.test_scores
        result &= self.ootsp_test_predictions == other.ootsp_test_predictions
        result &= self.ootsp_test_scores == other.ootsp_test_scores
        return result


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
        test_scores, ootsp_test_predictions, ootsp_test_predict_time, ootsp_test_scores
    )


def handle_evaluate(model_config: typing.Dict, arguments: argparse.Namespace):
    run_id = str(uuid.uuid4())

    output_dir = arguments.output_dir
    model_output_dir = None
    plot_dir = None
    if output_dir is not None:
        output_dir = os.path.join(getattr(arguments, 'output_dir'), run_id)
        model_output_dir = os.path.join(output_dir, 'model')
        os.makedirs(model_output_dir)
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir)

        record_run(run_id, output_dir, arguments=arguments, model_config=model_config)

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

    model_name = getattr(arguments, 'model')
    model = get_model(model_name, model_config, seed=getattr(arguments, 'model_seed'))

    result_scores = []
    for problem_name in getattr(arguments, 'problem'):
        if arguments.verbose:
            print('{} {} {}'.format(model_name, problem_name, run_id))
        problem = get_problem(problem_name, **vars(arguments))
        evaluate_result = evaluate(
            problem, model, model_config, train_data, test_data, ootsp_test_data, verbose=arguments.verbose,
            model_output_dir=model_output_dir, plot_dir=plot_dir
        )
        result_scores.append({
            'problem_name': problem_name,
            'model_name': model_name,
            **evaluate_result.__dict__,
        })

        if arguments.verbose:
            results = evaluate_result.__dict__
            del results['train_predictions']
            del results['test_predictions']
            del results['ootsp_test_predictions']
            print(json.dumps(results, indent=4))
            print()

    if output_dir is not None:
        record_run(run_id, output_dir, arguments=arguments, model_config=model_config, scores=result_scores)

    if output_dir is not None:
        if not os.listdir(model_output_dir):
            os.rmdir(model_output_dir)
        if not os.listdir(plot_dir):
            os.rmdir(plot_dir)

    return result_scores


def evaluate_handler(arguments: argparse.Namespace):
    model_config_path = getattr(arguments, 'model_config_path', None)
    if model_config_path is None:
        model_config = {}
    else:
        with open(model_config_path) as f:
            model_config = json.load(f)

    handle_evaluate(model_config, arguments)


def configure_tuning_parser(parser):
    parser.add_argument(
        '--tuning-config-path', type=str, action='store', required=True,
        help='the directory to read in the tuning config'
    )
    parser.add_argument(
        '--objective', type=str, action='store', required=True,
        choices=['total_rmse', 'top_k_regret', 'spearman', 'ndcg', 'ndcg@k']
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
    if arguments.objective == 'total_rmse':
        score_problem = 'regression'
        score_path = ('test_scores', 'total_scores', 'total_rmse')
        minimize = True
    elif arguments.objective == 'top_k_regret':
        score_problem = 'rank'
        score_path = ('test_scores', 'mean_scores', 'mean_top_k_regrets', arguments.k)
        minimize = True
    elif arguments.objective == 'spearman':
        score_problem = 'rank'
        score_path = ('test_scores', 'mean_scores', 'spearman_correlation', 'mean')
        minimize = False
    elif arguments.objective == 'ndcg':
        score_problem = 'rank'
        score_path = ('test_scores', 'mean_scores', 'mean_ndcg')
        minimize = False
    elif arguments.objective == 'ndcg@k':
        score_problem = 'rank'
        score_path = ('test_scores', 'mean_scores', 'ndcg_over_k', arguments.k)
        minimize = False
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
            import traceback
            traceback.print_exc()
            raise e from e

    return objective, minimize


def tuning_handler(arguments: argparse.Namespace):
    # gather config files
    with open(arguments.model_config_path, 'r') as file:
        model_config = json.load(file)
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


def rescore_handler(arguments: argparse.Namespace):
    if arguments.results_path is not None:
        result_paths = [arguments.results_path]
    elif arguments.results_dir is not None:
        result_paths = get_result_paths_from_dir(arguments.results_dir)
    elif arguments.result_paths_csv is not None:
        result_paths = get_result_paths_from_csv(arguments.result_paths_csv)

    for results_path in result_paths:
        handle_rescore(results_path, arguments.output_dir)


def handle_rescore(results_path: str, output_dir: str):
    with open(results_path) as f:
        results = json.load(f)

    train_data, test_data = get_train_and_test_data(
        results['arguments']['train_path'], results['arguments']['test_path'], results['arguments']['test_size'],
        results['arguments']['split_seed'], results['arguments']['metafeature_subset'],
        results['arguments']['cache_dir'], results['arguments']['no_cache']
    )

    # Create the output directory for the results of the re-score
    output_dir = os.path.join(output_dir, results['id'])
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    plot_dir = os.path.join(output_dir, 'plots')
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    # For each problem, re-score using the predictions and data and ensure the scores are the same as before
    for problem_scores in results['scores']:
        problem = get_problem(problem_scores['problem_name'], **results['arguments'])
        if problem is None:
            continue

        train_predictions = problem_scores['train_predictions']
        train_rescores = problem.score(train_predictions, train_data)
        problem_scores['train_scores'] = train_rescores
        problem.plot(train_predictions, train_data, train_rescores, os.path.join(plot_dir, 'train'))

        test_predictions = problem_scores['test_predictions']
        test_rescores = problem.score(test_predictions, test_data)
        problem_scores['test_scores'] = test_rescores
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
        train_data, test_data = split_data_by_group(train_data, 'dataset_id', test_size, split_seed)
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
    run_id: str, output_dir: str, *, arguments: argparse.Namespace, model_config: typing.Dict,
    scores: typing.Dict = None
):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, 'run.json')

    run = {
        'id': run_id,
        'git_commit': utils.get_git_commit_hash(),
        'arguments': arguments.__dict__,
        'model_config': model_config,
    }
    if scores is not None:
        run['scores'] = scores

    with open(path, 'w') as f:
        json.dump(run, f, indent=4, sort_keys=True)


def configure_report_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--results-dir', type=str, help='directory containing results'
    )
    parser.add_argument(
        '--result-paths-csv', type=str, help='path to csv containing paths to results'
    )
    parser.add_argument(
        '--report-dir', type=str, default='./leaderboards', help='directory to output score reports'
    )


def report_handler(arguments: argparse.Namespace):
    if arguments.results_dir is not None:
        result_paths = get_result_paths_from_dir(arguments.results_dir)
    elif arguments.result_paths_csv is not None:
        result_paths = get_result_paths_from_csv(arguments.result_paths_csv)
    else:
        raise ValueError('one of --results-dir or result-paths-csv must be provided')

    regression_results, rank_results, subset_results = load_results(result_paths)

    regression_leaderboard = make_leaderboard(regression_results, 'model_name', 'test.total_rmse', min)
    rank_leaderboard = make_leaderboard(rank_results, 'model_name', 'test.ndcg', max)
    subset_leaderboard = make_leaderboard(subset_results, 'model_name', 'test.ndcg_at_k.mean', max)

    if not os.path.isdir(arguments.report_dir):
        os.makedirs(arguments.report_dir)

    path = os.path.join(arguments.report_dir, 'regression_leaderboard.csv')
    regression_leaderboard.to_csv(path, index=False)
    path = os.path.join(arguments.report_dir, 'rank_leaderboard.csv')
    rank_leaderboard.to_csv(path, index=False)
    path = os.path.join(arguments.report_dir, 'subset_leaderboard.csv')
    subset_leaderboard.to_csv(path, index=False)

    save_result_paths_csv(regression_results, rank_results, subset_results, report_dir=arguments.report_dir)


def get_result_paths_from_dir(results_dir: str):
    return get_result_paths([os.path.join(results_dir, dir_) for dir_ in os.listdir(results_dir)])


def get_result_paths_from_csv(csv_path: str):
    df = pd.read_csv(csv_path, header=None)
    return get_result_paths(df[0])


def get_result_paths(result_dirs: typing.Sequence[str]):
    result_paths = []
    for dir_ in result_dirs:
        path = os.path.join(dir_, 'run.json')
        if os.path.isfile(path):
            result_paths.append(path)
        else:
            print(f'file does not exist: {path}')
    return result_paths


def load_results(result_paths: typing.Sequence[str]):
    regression_results = []
    rank_results = []
    subset_results = []
    for result_path in result_paths:
        result = load_result(result_path)
        if result == {}:
            print(f'no results for {result_path}')
        else:
            if 'regression' in result:
                regression_results.append(result['regression'])
            if 'rank' in result:
                rank_results.append(result['rank'])
            if 'subset' in result:
                subset_results.append(result['subset'])
    return pd.DataFrame(regression_results), pd.DataFrame(rank_results), pd.DataFrame(subset_results)


def load_result(result_path: str):
    with open(result_path) as f:
        run = json.load(f)

    result = {}
    for problem_scores in run.get('scores', []):
        problem_name = problem_scores['problem_name']
        if problem_name == 'regression':
            parsed_scores = parse_regression_scores(problem_scores)
        elif problem_name == 'rank':
            parsed_scores = parse_rank_scores(problem_scores)
        elif problem_name == 'subset':
            parsed_scores = parse_subset_scores(problem_scores)
        parsed_scores['path'] = os.path.dirname(result_path)
        result[problem_name] = parsed_scores
    return result


def parse_regression_scores(scores: typing.Dict):

    def _parse(scores, parent_key):
        return {
            **utils.flatten(scores['mean_scores'], parent_key),
            **utils.flatten(scores['total_scores'], parent_key),
        }

    return {
        'model_name': scores['model_name'],
        **_parse(scores['train_scores'], 'train'),
        **_parse(scores['test_scores'], 'test')
    }


def parse_rank_scores(scores: typing.Dict):
    return {
        'model_name': scores['model_name'],
        **utils.flatten(scores['train_scores']['mean_scores'], 'train'),
        **utils.flatten(scores['test_scores']['mean_scores'], 'test'),
    }


def parse_subset_scores(scores: typing.Dict):
    return {
        'model_name': scores['model_name'],
        **utils.flatten(scores['train_scores'], 'train'),
        **utils.flatten(scores['test_scores'], 'test'),
    }


def make_leaderboard(results: pd.DataFrame, id_col: str, score_col: str, opt: typing.Callable):
    grouped_results = results.groupby(id_col)
    counts = grouped_results.size().astype(int)
    counts.name = 'count'
    leader_indices = grouped_results[score_col].transform(opt) == results[score_col]
    leaderboard = results[leader_indices].drop_duplicates([id_col, score_col])
    leaderboard.sort_values([score_col, id_col], ascending=opt==min, inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)
    leaderboard = leaderboard.join(counts, on=id_col)

    columns = list(leaderboard.columns)
    columns.remove(id_col)
    columns.remove(score_col)
    columns.remove(counts.name)

    leaderboard = leaderboard[[id_col, counts.name, score_col] + sorted(columns)]

    return leaderboard.round(8)


def save_result_paths_csv(*args: pd.DataFrame, report_dir):
    result_paths = []
    for results in args:
        result_paths += results['path'].tolist()
    df = pd.DataFrame(set(result_paths))
    path = os.path.join(report_dir, 'result_paths.csv')
    df.to_csv(path, header=False, index=False)


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

    arguments = parser.parse_args(argv[1:])

    handler(arguments, parser)


if __name__ == '__main__':
    main(sys.argv)
