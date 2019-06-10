import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import mean_squared_error
import utils


def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    return np.sum(y_hat == y, dtype=np.float32) / len(y)


def rmse(y_hat, y):
    """
    Calculates the unbiased standard deviation of the residuals.
    """
    return mean_squared_error(np.array(y_hat), np.array(y))**.5


def top_k_correct(ranked_data: dict, actual_data: dict, k):
    ranked_df = pd.DataFrame(ranked_data)
    actual_df = pd.DataFrame(actual_data)
    top_actual = [pipeline["id"] for pipeline in actual_df.nlargest(k, columns='test_f1_macro').pipeline]
    top_predicted = ranked_df.nsmallest(k, columns="rank").pipeline_id
    return len(set(top_actual).intersection(set(top_predicted)))


def top_k_regret(ranked_data, actual_data, k):
    pipeline_ids = pd.Series([instance['pipeline']['id'] for instance in actual_data], name='pipeline_id')
    actual_df = pd.concat([pd.DataFrame(actual_data), pipeline_ids], axis=1)
    actual_best_score = actual_df['test_f1_macro'].max()

    top_k_ranked = pd.DataFrame(ranked_data).nsmallest(k, columns="rank").pipeline_id
    min_regret = np.inf
    for index, pipeline_id in top_k_ranked.iteritems():
        regret = actual_best_score - actual_df[actual_df['pipeline_id'] == pipeline_id]['test_f1_macro'].iloc[0]
        min_regret = min(min_regret, regret)
    return min_regret


def spearman_correlation(ranked_data, actual_data):
    actual_data = pd.DataFrame(actual_data)
    score = scipy.stats.spearmanr(ranked_data['rank'], utils.rank(actual_data.test_f1_macro))
    return score.correlation
