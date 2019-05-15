import numpy as np


def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    return np.sum(y_hat == y, dtype=np.float32) / len(y)


def rmse(y_hat, y):
    """
    Calculates the unbiased standard deviation of the residuals.
    """
    return np.std(np.array(y_hat) - np.array(y), ddof=1)
