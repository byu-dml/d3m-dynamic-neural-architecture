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

def top_k(ranked_data: dict, actual_data: dict, k):
    ranked_df = pd.DataFrame(ranked_data)
    actual_df = pd.DataFrame(actual_data)
    top_actual = [pipeline["id"] for pipeline in actual_df.nlargest(k, columns='test_f1_macro').pipeline]
    top_predicted = ranked_df.nsmallest(k, columns="rank").pipeline_id
    return len(set(top_actual).intersection(set(top_predicted)))

def regret_value(ranked_data, actual_data):
    actual_df = pd.DataFrame(actual_data)
    actual_best_metric_value = actual_df['test_f1_macro'].max()  # np.nanmax

    best_ranked_index = np.argmin(ranked_data['rank'])
    best_ranked_pipeline_id = ranked_data['pipeline_id'][best_ranked_index]
    for instance in actual_data:
        if instance['pipeline']['id'] == best_ranked_pipeline_id:
            return actual_best_metric_value - instance['test_f1_macro']

def spearman_correlation(ranked_data, actual_data):
    actual_data = pd.DataFrame(actual_data)
    score = scipy.stats.spearmanr(ranked_data['rank'], utils.rank(actual_data.test_f1_macro))
    return score.correlation
