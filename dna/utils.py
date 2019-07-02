import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

def rank(values):
    return len(values) - 1 - np.argsort(values)


def plot_spearman(ranked_data: pd.DataFrame, actual_data: pd.DataFrame, name: str, directory: str, score: float):
    actual_data = pd.DataFrame(actual_data)
    ranked_data = ranked_data['rank']
    actual_data = rank(actual_data.test_f1_macro)

    plt.title('Spearman Correlation: ' + name + ' = ' + str(score))
    plt.xlabel('Ranked Data')
    plt.ylabel('Actual Data')
    plt.scatter(ranked_data, actual_data)
    new_dir = os.path.join(directory, 'spearman_plots')
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    file_name = os.path.join(new_dir, name)
    plt.savefig(fname=file_name)
    plt.clf()


def plot_pearson(y_hat, y, name: str, directory: str, score: float):
    plt.title('Pearson Correlation: ' + name + ' = ' + str(score))
    plt.xlabel('y_hat')
    plt.ylabel('y')
    plt.scatter(y_hat, y)
    new_dir = os.path.join(directory, 'pearson_plots')
    if not os.path.isdir(new_dir):
        os.makedirs(new_dir)
    file_name = os.path.join(new_dir, name)
    plt.savefig(fname=file_name)
    plt.clf()
