import matplotlib.pyplot as plt
import pandas as pd


def plot_at_k_scores(model_names: pd.Series, model_scores, model_colors, plot_path, ylabel=None, title=None, max_k=100):
    for model_name, at_k_scores, model_color in zip(model_names, model_scores, model_colors):
        plt.plot(range(1, max_k + 1), at_k_scores[:max_k], label=model_name, color=model_color)
    plt.xlabel('k')
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend(loc=0)
    plt.savefig(plot_path)
    plt.clf()
