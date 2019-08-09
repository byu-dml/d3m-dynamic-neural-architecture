import os
import json
import argparse
import numpy as np
import pandas as pd


def load_results(results_dir_: str):
    results_path = os.path.join(results_dir_, 'run.json')
    with open(results_path) as f:
        results_ = json.load(f)
    return results_


def append_score(score_list: list, path_idx_list: list, score: float, idx: int):
    score_list.append(score)
    path_idx_list.append(idx)


def print_best_score(score_list: list, path_idx_list: list, best_function, score_name: str, paths_: list):
    best_score_idx = best_function(score_list)
    best_score = score_list[best_score_idx]
    best_path_idx = path_idx_list[best_score_idx]
    best_path = paths_[best_path_idx]
    print('Best {0} of {1} found at {2}'.format(score_name, best_score, best_path))


parser = argparse.ArgumentParser()
parser.add_argument('--results-dir', type=str, default='results')
parser.add_argument('--csv', type=str, default=None)
args = parser.parse_args()


if args.csv is not None:
    df = pd.read_csv(args.csv)
    paths = df.Path
    total_rmse_scores = []
    total_rmse_path_indices = []
    mean_spearman_scores = []
    mean_spearman_path_indices = []
    top_k_regret_scores = []
    top_k_regret_path_indices = []

    for path_idx, results_dir in enumerate(paths):
        results = load_results(results_dir)
        problem_scores = results['scores']
        for scores in problem_scores:
            problem_name = scores['problem_name']
            test_scores = scores['test_scores']
            if problem_name == 'regression':
                total_rmse = test_scores['total_scores']['total_rmse']
                append_score(total_rmse_scores, total_rmse_path_indices, total_rmse, path_idx)
            elif problem_name == 'rank':
                mean_spearman = test_scores['mean_scores']['spearman_correlation']['mean']
                append_score(mean_spearman_scores, mean_spearman_path_indices, mean_spearman, path_idx)
            elif problem_name == 'subset':
                top_k_regret = test_scores['top_k_regret']['mean']
                append_score(top_k_regret_scores, top_k_regret_path_indices, top_k_regret, path_idx)

    print_best_score(total_rmse_scores, total_rmse_path_indices, np.argmin, 'TOTAL RMSE', paths)
    print_best_score(mean_spearman_scores, mean_spearman_path_indices, np.argmax, 'MEAN SPEARMAN', paths)
    print_best_score(top_k_regret_scores, top_k_regret_path_indices, np.argmin, 'TOP K REGRET', paths)
else:
    results_dir = args.results_dir
    dirs = os.listdir(results_dir)

    for uuid_dir in dirs:
        results_uuid_dir = os.path.join(results_dir, uuid_dir)
        results = load_results(results_uuid_dir)
        id_ = results['id']
        try:
            print('ID:', id_)
            problem_scores = results['scores']
            for scores in problem_scores:
                problem_name = scores['problem_name']
                train_scores = scores['train_scores']
                test_scores = scores['test_scores']
                train_scores = train_scores['mean_scores'] if problem_name != 'subset' else train_scores
                test_scores = test_scores['mean_scores'] if problem_name != 'subset' else test_scores
                print('PROBLEM:', problem_name)
                print('MEAN TRAIN SCORES:', train_scores)
                print('MEAN TEST SCORES:', test_scores)
                print()
        except(KeyError, TypeError):
            print('Results at id: {0} do not have scores'.format(id_))
            print()
        print('#######################################################################################################')
        print()
