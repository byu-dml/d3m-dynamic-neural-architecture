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


def _get_relevance_at_k(relevance: typing.Sequence, rank: typing.Sequence, k: int):
    relevance = np.array(relevance)
    if rank is None:
        relevance_at_k = relevance[:k]
    else:
        rank_order = np.argsort(rank)
        relevance_at_k = relevance[rank_order[:k]]

    return relevance_at_k


def top_k_correct(relevance: typing.Sequence, rank: typing.Sequence, k: int = None):
    if k is None:
        k = len(relevance)

    k_best_indices = set(np.argsort(relevance)[-k:])
    k_ranked_indices = set(np.argsort(rank)[:k])
    return len(k_best_indices.intersection(k_ranked_indices))


def top_k_regret(relevance: typing.Sequence, rank: typing.Sequence = None, k: int = None):
    if k is None:
        k = len(relevance)

    best = max(relevance)
    relevance_at_k = _get_relevance_at_k(relevance, rank, k)
    return best - max(relevance_at_k)


def spearman_correlation(x: typing.Sequence, y: typing.Sequence):
    spearman = scipy.stats.spearmanr(x, y)
    return spearman.correlation, spearman.pvalue


def dcg_at_k(
    relevance: typing.Sequence, rank: typing.Sequence = None, k: int = None, gains_f: str = 'exponential'
):
    """Discounted cumulative gain (DCG) at rank k

    Parameters
    ----------
    relevance: Sequence
        True relevance labels
    rank: Sequence
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

    relevance_at_k = _get_relevance_at_k(relevance, rank, k)

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
