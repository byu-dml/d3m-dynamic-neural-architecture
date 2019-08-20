import argparse
import collections
import json
import os
import sys
import typing

import pandas as pd


def configure_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--results-dir', type=str, default=None, help='directory containing results'
    )
    parser.add_argument(
        '--result-paths-csv', type=str, default=None, help='path to csv containing paths to results'
    )
    parser.add_argument(
        '--report-dir', type=str, default='./leaderboards', help='directory to output score reports'
    )


def handler(arguments: argparse.Namespace):
    if arguments.results_dir is not None:
        result_paths = get_result_paths_from_dir(arguments.results_dir)
    elif arguments.result_paths_csv is not None:
        result_paths = get_result_paths_from_csv(arguments.result_paths_csv)
    else:
        raise ValueError('one of --results-dir or result-paths-csv must be provided')

    regression_results, rank_results, subset_results = load_results(result_paths)

    regression_leaderboard = make_leaderboard(regression_results, 'model_name', 'test.total_rmse', min)
    rank_leaderboard = make_leaderboard(rank_results, 'model_name', 'test.spearman_correlation.mean', max)
    subset_leaderboard = make_leaderboard(subset_results, 'model_name', 'test.top_k_regret.mean', min)

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


def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_regression_scores(scores: typing.Dict):

    def _parse(scores, parent_key):
        return {
            **flatten(scores['mean_scores'], parent_key),
            **flatten(scores['total_scores'], parent_key), 
        }

    return {
        'model_name': scores['model_name'],
        **_parse(scores['train_scores'], 'train'),
        **_parse(scores['test_scores'], 'test')
    }


def parse_rank_scores(scores: typing.Dict):
    return {
        'model_name': scores['model_name'],
        **flatten(scores['train_scores']['mean_scores'], 'train'),
        **flatten(scores['test_scores']['mean_scores'], 'test'), 
    }


def parse_subset_scores(scores: typing.Dict):
    return {
        'model_name': scores['model_name'],
        **flatten(scores['train_scores'], 'train'),
        **flatten(scores['test_scores'], 'test'), 
    }


def make_leaderboard(results: pd.DataFrame, id_col: str, score_col: str, opt: typing.Callable):
    leader_indices = results.groupby(id_col)[score_col].transform(opt) == results[score_col]
    leaderboard = results[leader_indices].drop_duplicates([id_col, score_col])
    leaderboard.sort_values([score_col, id_col], ascending=opt==min, inplace=True)
    leaderboard.reset_index(drop=True, inplace=True)
    return leaderboard.round(6)


def save_result_paths_csv(*args: pd.DataFrame, report_dir):
    result_paths = []
    for results in args:
        result_paths += results['path'].tolist()
    df = pd.DataFrame(set(result_paths))
    path = os.path.join(report_dir, 'result_paths.csv')
    df.to_csv(path, header=False, index=False)


def main(argv: typing.Sequence):
    parser = argparse.ArgumentParser()
    configure_parser(parser)
    arguments = parser.parse_args(argv[1:])
    handler(arguments)


if __name__ == '__main__':
    main(sys.argv)
