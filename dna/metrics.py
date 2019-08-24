import typing
import warnings

import pandas as pd
import numpy as np
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
    assert len(top_k_predicted) <= k, 'length of top_k_predicted ({}) is greater than k ({})'.format(len(top_k_predicted), k)
    top_actual = actual_data.nlargest(k, columns='test_f1_macro', keep='all')['pipeline_id']
    return len(set(top_actual).intersection(set(top_k_predicted)))


def top_k_regret(top_k_predicted: typing.Sequence, actual_data: pd.DataFrame, k: int):
    """
    Assumes that the top_k_predicted list is sorted.
    """
    top_k_predicted = top_k_predicted[:k]
    assert len(top_k_predicted) <= k, 'length of top_k_predicted ({}) is greater than k ({})'.format(len(top_k_predicted), k)
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


def dcg_at_k(
    relevance: typing.Sequence, rank: typing.Sequence = None, k: int = None, gains_f: str = 'exponential'
):
    """Discounted cumulative gain (DCG) at rank k

    Parameters
    ----------
    actual_relevance: Sequence
        True relevance labels
    predicted_rank: Sequence
        Predicted rank for actual_relevance. If not provided, actual_relevance is assumed to be in rank order.
    k: int
        Rank position.
    gains: str
        Whether gains should be "exponential" (default) or "linear".
    idcg: bool
        the ideal ordering of the dcg, used for calculating the ndcg

    Returns
    -------
    DCG@k: float
    """
    if k is None:
        k = len(relevance)

    relevance = np.array(relevance)
    if rank is None:
        relevance_at_k = relevance[:k]
    else:
        rank_order = np.argsort(rank)
        relevance_at_k = relevance[rank_order[:k]]

    if gains_f == 'exponential':
        gains = 2 ** relevance_at_k - 1
    elif gains_f == 'linear':
        gains = relevance_at_k
    else:
        raise ValueError('Invalid gains_f: {}'.format(gains_f))

    # discount = log2(i + 1), with i starting at 1
    discounts = np.log2(np.arange(2, k + 2))
    return np.sum(gains / discounts)


def ndcg_at_k(
    relevance: typing.Sequence, rank: typing.Sequence = None, k: int = None, gains_f: str = 'exponential'
):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    relevance: Sequence
        True relevance labels
    rank: Sequence
        Predicted rank for relevance. If not provided, relevance is assumed to be in rank order.
    k: int
        Rank position.
    gains: str
        Whether gains should be "exponential" (default) or "linear".

    Returns
    -------
    NDCG@k: float
    """
    dcg = dcg_at_k(relevance, rank, k=k, gains_f=gains_f)
    idcg = dcg_at_k(np.sort(relevance)[::-1], k=k, gains_f=gains_f)
    return dcg / idcg


def average_precision(y_true, y_score, k):
    """
    Average precision of rankings

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth of the top pipeline ids
    y_score : array, shape = [n_samples]
        Predicted top ids
    k :  the first k pipelines that we will compare from each list

    Returns
    -------
    score : float, 1 is perfect, 0 is completely wrong

    Mean average precision will be calculated after all the average precisions are calculated in `problems.py` since it is just a simple mean
    """
    assert len(y_true) != 0 and k != 0 and len(y_score) != 0, 'given a zero length input'
    # truncate to k piplines
    y_true = y_true[:k]
    y_score = y_score[:k]
    assert len(y_true) == len(y_score), 'given arrays for AP were not the same length: {} vs {}'.format(len(y_true), len(y_score))
    current_set = set()
    relevance_mask = []
    precision_list = []
    for index, id_value in enumerate(y_score):
        k = index + 1
        current_set.add(y_true[index])
        relevance_mask.append(1 if id_value in current_set else 0)
        precision_at_k = len(set(y_score[:k]).intersection(current_set)) / k
        precision_list.append(precision_at_k)

    precision = np.dot(np.array(relevance_mask), np.array(precision_list))
    return precision / k
