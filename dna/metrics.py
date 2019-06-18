import warnings

import pandas as pd
import scipy.stats
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pyplot as plt
import os

import utils


def accuracy(y_hat, y):
    return accuracy_score(y, y_hat)


def rmse(y_hat, y):
    return mean_squared_error(y, y_hat)**.5


def pearson_correlation(y_hat, y, name: str, directory: str):
    """
    Calculates Pearson's R^2 coefficient. Returns a tuple containing the correlation coefficient and the p value for
    the test that the correlation coefficient is different than 0.
    """
    plt.title('Pearson Correlation: ' + name)
    plt.xlabel('y_hat')
    plt.ylabel('y')
    plt.scatter(y_hat, y)
    new_dir = os.path.join(directory, 'pearson_plots')
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    file_name = os.path.join(new_dir, name)
    plt.savefig(fname=file_name)
    plt.clf()

    with warnings.catch_warnings(record=True) as w:
        return scipy.stats.pearsonr(y_hat, y)


def top_k_correct(ranked_data: pd.DataFrame, actual_data: pd.DataFrame, k: int):
    top_actual = [pipeline["id"] for pipeline in actual_data.nlargest(k, columns='test_f1_macro').pipeline]
    top_predicted = ranked_data.nsmallest(k, columns="rank").pipeline_id
    return len(set(top_actual).intersection(set(top_predicted)))


def top_k_regret(ranked_data: pd.DataFrame, actual_data: pd.DataFrame, k: int):
    pipeline_ids = pd.Series([pipeline['id'] for pipeline in actual_data['pipeline']], name='pipeline_id')
    actual_df = pd.concat([pd.DataFrame(actual_data), pipeline_ids], axis=1)
    actual_best_score = actual_df['test_f1_macro'].max()

    top_k_ranked = ranked_data.nsmallest(k, columns="rank").pipeline_id
    min_regret = float('inf')
    for index, pipeline_id in top_k_ranked.iteritems():
        regret = actual_best_score - actual_df[actual_df['pipeline_id'] == pipeline_id]['test_f1_macro'].iloc[0]
        min_regret = min(min_regret, regret)
    return min_regret


def spearman_correlation(ranked_data: pd.DataFrame, actual_data: pd.DataFrame, name: str, directory: str):
    actual_data = pd.DataFrame(actual_data)
    ranked_data = ranked_data['rank']
    actual_data = utils.rank(actual_data.test_f1_macro)
    score = scipy.stats.spearmanr(ranked_data, actual_data)

    plt.title('Spearman Correlation: ' + name)
    plt.xlabel('Ranked Data')
    plt.ylabel('Actual Data')
    plt.scatter(ranked_data, actual_data)
    new_dir = os.path.join(directory, 'spearman_plots')
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    file_name = os.path.join(new_dir, name)
    plt.savefig(fname=file_name)
    plt.clf()

    return score.correlation
