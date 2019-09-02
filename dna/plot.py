import matplotlib.pyplot as plt
import pandas as pd


def plot_at_k_scores(model_names: pd.Series, model_scores, plot_path, ylabel, title, max_k=100):
    for model_name, at_k_scores in zip(model_names, model_scores):
        if model_name in [
            'attention_regression','daglstm_regression','lstm','dna_regression','dag_attention_regression',
            'autosklearn','meta_autosklearn','random_forest','linear_regression',
        ]:
            plt.plot(range(1, max_k + 1), at_k_scores[:max_k], label=model_name)
    plt.xlabel('k')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=0)
    plt.savefig(plot_path)
    plt.clf()
