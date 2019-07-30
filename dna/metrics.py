import typing
import warnings

import pandas as pd
import scipy.stats
from sklearn.metrics import accuracy_score, mean_squared_error

from dna import utils


def accuracy(y_hat, y):
    return accuracy_score(y, y_hat)


def rmse(y_hat, y):
    return mean_squared_error(y, y_hat)**.5


def pearson_correlation(y_hat, y):
    """
    Calculates Pearson's R^2 coefficient. Returns a tuple containing the correlation coefficient and the p value for
    the test that the correlation coefficient is different than 0.
    """
    with warnings.catch_warnings(record=True) as w:
        return scipy.stats.pearsonr(y_hat, y)


def top_k_correct(top_k_predicted: typing.Sequence, actual_data: pd.DataFrame, k: int):
    """
    Assumes that the top_k_predicted list is sorted. 
    """
    top_k_predicted = top_k_predicted[:k]
    assert len(top_k_predicted) <= k, "The number of pipelines given is more than K: {} vs {}".format(len(top_k_predicted), k)
    top_actual = actual_data.nlargest(k, columns='test_f1_macro', keep='all')['pipeline_id']
    return len(set(top_actual).intersection(set(top_k_predicted)))


def top_k_regret(top_k_predicted: typing.Sequence, actual_data: pd.DataFrame, k: int):
    """
    Assumes that the top_k_predicted list is sorted. 
    """
    top_k_predicted = top_k_predicted[:k]
    assert len(top_k_predicted) <= k, "The number of pipelines given is more than K: {} vs {}".format(len(top_k_predicted), k)
    actual_best_score = actual_data['test_f1_macro'].max()
    min_regret = float('inf')
    for pipeline_id in top_k_predicted:
        pipeline_score = actual_data[actual_data['pipeline_id'] == pipeline_id]['test_f1_macro'].iloc[0]
        regret = actual_best_score - pipeline_score
        min_regret = min(min_regret, regret)
    return min_regret


def spearman_correlation(x: typing.Sequence, y: typing.Sequence):
    spearman = scipy.stats.spearmanr(x, y)
    return spearman.correlation, spearman.pvalue
