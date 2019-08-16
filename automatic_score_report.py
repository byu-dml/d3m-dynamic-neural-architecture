import os
import json
import argparse
import numpy as np
import pandas as pd


def load_results(results_dir: str):
    results_path = os.path.join(results_dir, 'run.json')
    with open(results_path) as f:
        results = json.load(f)
    return results


def get_regression_dict():
    return {
        'Model': [],
        'Total RMSE': [],
        'Total Pearson Coeff.': [],
        'Total Pearson p-value': [],
        'Mean RMSE': [],
        'RMSE SD': [],
        'Mean Pearson Coeff.': [],
        'SD Pearson Coeff.': [],
        'Mean Pearson P-value': [],
        'SD Pearson P-value': [],
        'Fit Time': [],
        'Test Predict Time': [],
        'Path': []
    }


def get_rank_dict():
    return {
        'Model': [],
        'Spearman Mean': [],
        'Spearman SD': [],
        'Spearman P': [],
        'Spearman P SD': [],
        'Fit Time': [],
        'Test Predict Time': [],
        'Path': []
    }


def get_subset_dict():
    return {
        'Model': [],
        'Top 1 Regret Mean': [],
        'Top 1 SD': [],
        'Top 25 Regret': [],
        'Top 25 SD': [],
        'Top 25 Count': [],
        'Top 25 Count SD': [],
        'Fit Time': [],
        'Test Predict Time': [],
        'Path': []
    }


def add_vals(problem_dict: dict, scores: dict, path: str):
    problem_dict['Model'].append(scores['model_name'])
    problem_dict['Fit Time'].append(scores['fit_time'])
    problem_dict['Test Predict Time'].append(scores['test_predict_time'])
    problem_dict['Path'].append(path)

def add_regression_vals(regression_dict: dict, scores: dict, path: str):
    add_vals(regression_dict, scores, path)

    test_scores = scores['test_scores']
    mean_scores = test_scores['mean_scores']
    mean_pearson = mean_scores['pearson_correlation']
    mean_rmse = mean_scores['rmse']
    total_scores = test_scores['total_scores']
    total_pearson = total_scores['total_pearson_correlation']
    total_rmse = total_scores['total_rmse']

    regression_dict['Total RMSE'].append(total_rmse)
    regression_dict['Total Pearson Coeff.'].append(total_pearson['correlation_coefficient'])
    regression_dict['Total Pearson p-value'].append(total_pearson['p_value'])
    regression_dict['Mean RMSE'].append(mean_rmse['mean'])
    regression_dict['RMSE SD'].append(mean_rmse['std_dev'])
    regression_dict['Mean Pearson Coeff.'].append(mean_pearson['mean'])
    regression_dict['SD Pearson Coeff.'].append(mean_pearson['std_dev'])
    regression_dict['Mean Pearson P-value'].append(mean_pearson['mean_p_value'])
    regression_dict['SD Pearson P-value'].append(mean_pearson['std_dev_p_value'])


def add_rank_vals(rank_dict: dict, scores: dict, path: str):
    add_vals(rank_dict, scores, path)

    mean_spearman_scores = scores['test_scores']['mean_scores']['spearman_correlation']
    rank_dict['Spearman Mean'].append(mean_spearman_scores['mean'])
    rank_dict['Spearman SD'].append(mean_spearman_scores['std_dev'])
    rank_dict['Spearman P'].append(mean_spearman_scores['mean_p_value'])
    rank_dict['Spearman P SD'].append(mean_spearman_scores['std_dev_p_value'])


def add_subset_vals(subset_dict: dict, scores: dict, path: str):
    add_vals(subset_dict, scores, path)

    test_scores = scores['test_scores']
    top_1_regret = test_scores['top_1_regret']
    top_k_regret = test_scores['top_k_regret']
    top_k_count = test_scores['top_k_count']

    subset_dict['Top 1 Regret Mean'].append(top_1_regret['mean'])
    subset_dict['Top 1 SD'].append(top_1_regret['std_dev'])
    subset_dict['Top 25 Regret'].append(top_k_regret['mean'])
    subset_dict['Top 25 SD'].append(top_k_regret['std_dev'])
    subset_dict['Top 25 Count'].append(top_k_count['mean'])
    subset_dict['Top 25 Count SD'].append(top_k_count['std_dev'])


def save_leader_board_csv(csv_path: str, problem_dict: dict, col_name: str, argmax: bool = False):
   problem_df = pd.DataFrame(problem_dict)
   model_names = list(problem_df['Model'])
   tmp = set()
   tmp.update(model_names)
   model_names = sorted(tmp)

   df_to_save = None
   for model_name in model_names:
       # Get the rows that correspond to the current model
       model_indices = problem_df['Model'] == model_name
       model_df = problem_df[model_indices]

       # Get the model with the best performance
       best_idx = model_df[col_name].idxmax() if argmax else model_df[col_name].idxmin()
       best_model_row = model_df.loc[[best_idx]] # Put index in a list in order to get a data frame instead of a series
       if df_to_save is None:
           df_to_save = best_model_row
       else:
            df_to_save = pd.concat([df_to_save, best_model_row])

   df_to_save.to_csv(csv_path)


def append_score(score_list: list, path_idx_list: list, score: float, idx: int):
    score_list.append(score)
    path_idx_list.append(idx)


def print_best_score(score_list: list, path_idx_list: list, best_function, score_name: str, paths_: list):
    best_score_idx = best_function(score_list)
    best_score = score_list[best_score_idx]
    best_path_idx = path_idx_list[best_score_idx]
    best_path = paths_[best_path_idx]
    print('Best {0} of {1} found at {2}'.format(score_name, best_score, best_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--csv-to-load', type=str, default=None,
                        help='Path where the csv of results directories is stored')
    parser.add_argument('--regression-csv', type=str, default=None,
                        help='Path where the regression leader board csv is saved')
    parser.add_argument('--rank-csv', type=str, default=None,
                        help='Path where the rank leader board csv is saved')
    parser.add_argument('--subset-csv', type=str, default=None,
                        help='Path where the subset leader board csv is saved')
    args = parser.parse_args()


    csv_to_load = args.csv_to_load
    if csv_to_load is not None:
        df = pd.read_csv(csv_to_load)
        paths = df.Path
        regression_dict = get_regression_dict()
        rank_dict = get_rank_dict()
        subset_dict = get_subset_dict()
        total_rmse_scores = []
        total_rmse_path_indices = []
        mean_spearman_scores = []
        mean_spearman_path_indices = []
        top_k_regret_scores = []
        top_k_regret_path_indices = []

        for path_idx, results_dir in enumerate(paths):
            results = load_results(results_dir)

            try:
                problem_scores = results['scores']
            except(KeyError):
                print('The run at {} has no scores'.format(results_dir))

            for scores in problem_scores:
                problem_name = scores['problem_name']
                test_scores = scores['test_scores']
                if problem_name == 'regression':
                    add_regression_vals(regression_dict, scores, results_dir)

                    total_rmse = test_scores['total_scores']['total_rmse']
                    append_score(total_rmse_scores, total_rmse_path_indices, total_rmse, path_idx)
                elif problem_name == 'rank':
                    add_rank_vals(rank_dict, scores, results_dir)

                    mean_spearman = test_scores['mean_scores']['spearman_correlation']['mean']
                    append_score(mean_spearman_scores, mean_spearman_path_indices, mean_spearman, path_idx)
                elif problem_name == 'subset':
                    add_subset_vals(subset_dict, scores, results_dir)

                    top_k_regret = test_scores['top_k_regret']['mean']
                    append_score(top_k_regret_scores, top_k_regret_path_indices, top_k_regret, path_idx)

        # Save the leader boards as csv files
        save_leader_board_csv(args.regression_csv, regression_dict, 'Total RMSE')
        save_leader_board_csv(args.rank_csv, rank_dict, 'Spearman Mean', argmax=True)
        save_leader_board_csv(args.subset_csv, subset_dict, 'Top 25 Regret')

        # Print the best of the best
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


if __name__ == '__main__':
    main()
