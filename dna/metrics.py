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


def spearman_correlation(ranked_data: pd.DataFrame, actual_data: pd.DataFrame):
    actual_data = pd.DataFrame(actual_data)
    score = scipy.stats.spearmanr(ranked_data['rank'], utils.rank(actual_data.test_f1_macro))
    return score.correlation, score.pvalue
