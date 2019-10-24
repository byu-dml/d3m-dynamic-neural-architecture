import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def plot_at_k_scores_over_k(
    model_names: pd.Series, model_scores, model_errors, model_colors, plot_path, ylabel=None, title=None, max_k=100
):
    for model_name, at_k_scores, at_k_errors, model_color in zip(model_names, model_scores, model_errors, model_colors):
        x = np.arange(1, max_k + 1)
        y = np.array(at_k_scores[:max_k])
        yerr = np.array(at_k_errors[:max_k])
        plt.plot(x, y, label=model_name, color=model_color)
        plt.fill_between(x, y-yerr, y+yerr, facecolor=model_color, edgecolor=None, alpha=.25)

    plt.xlabel('k')
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend(loc=0)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.clf()


def plot_violin_of_score_distributions(scores_by_model, model_colors, ylabel, title, plot_path):
    data = pd.DataFrame.from_dict(scores_by_model, orient='index').T
    ax = sns.violinplot(data=data, palette=model_colors, cut=0)
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
