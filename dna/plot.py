import collections
import colorsys
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from dna.models import get_model_class


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
    plt.savefig(plot_path)
    plt.clf()


def plot_violin_of_score_distributions(scores_by_model, score_name, model_colors, plot_path):
    # todo: use model names and colors
    ax = sns.violinplot(data=pd.DataFrame(scores_by_model), cut=0, palette=model_colors)
    plt.title('Distribution of {}'.format(score_name))
    plt.xlabel('Model')
    plt.ylabel('{}'.format(score_name))
    plt.savefig(plot_path)
    plt.close()


def create_distribution_plots(results: pd.DataFrame, output_dir: str, list_of_k: list):
    scores_by_metric_by_model = {}

    def insert_score(metric_name, model_id, score_value):
        nonlocal scores_by_metric_by_model

        if metric_name not in scores_by_metric_by_model:
            scores_by_metric_by_model[metric_name] = {}

        if model_id not in scores_by_metric_by_model[metric_name]:
            scores_by_metric_by_model[metric_name][model_id] = []

        if type(score_value) == list:
            scores_by_metric_by_model[metric_name][model_id].extend(score_value)
        else:
            scores_by_metric_by_model[metric_name][model_id].append(score_value)

    model_colors = {}

    for row_number, row in results.iterrows():
        model_id = row['model_id']
        model_class = get_model_class(model_id)
        model_colors[model_class.name] = model_class.color
        for column_name, cell_value in row.iteritems():
            if column_name.startswith('test.scores_by_dataset_id'):
                assert column_name.endswith('_by_run')

                _, _, dataset_id, raw_metric_name = column_name.split('.')
                metric_name = raw_metric_name.replace('_by_run', '')

                if 'at_k' in metric_name:
                    for k in list_of_k:
                        pretty_metric_name = metric_name.replace('k', str(k) if k != -1 else 'Max_k')
                        assert type(cell_value) == list and type(cell_value[0]) == list
                        for run_scores_at_k in cell_value:
                            if k <= len(run_scores_at_k):
                                insert_score(pretty_metric_name, model_class.name, run_scores_at_k[k-1]) # index 0 contains values for k=1

                else:
                    insert_score(metric_name, model_class.name, cell_value)

    for metric_name, scores_by_model_id in scores_by_metric_by_model.items():
        plot_violin_of_score_distributions(
            scores_by_model_id, metric_name, model_colors,
            plot_path=os.path.join(output_dir, '{}_distributions.pdf'.format(metric_name))
        )
