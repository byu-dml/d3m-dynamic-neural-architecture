import numpy as np


def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    return np.sum(y_hat == y, dtype=np.float32) / len(y)


def rmse(y_hat, y):
    """
    Calculates the unbiased standard deviation of the residuals.
    """
    return np.std(np.array(y_hat) - np.array(y), ddof=1)

def top_k(ranked_df, actual_df, k):
    """
    A metric for calculating how many of the predicted top K pipelines are actually in the real top k
    :param ranked_df:
    :param actual_df:
    :param k: the number of top pipelines to compare with
    :return:
    """
    top_actual = actual_df.nlargest(k, columns="score").id
    top_predicted = ranked_df.nlargest(k, columns="score").id
    return len(set(top_actual).intersection(set(top_predicted)))

def regret_value(ranked_df, actual_df):
    """

    :param ranked_df:  a Pandas DF with columns id (for pipeline), score
    :param actual_df:
    :return:
    """
    opt = np.nanmax
    best_metric_value = opt(actual_df["score"])
    best_predicted_value = opt(ranked_df["score"])
    return abs(best_metric_value - best_predicted_value)
