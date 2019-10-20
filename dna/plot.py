import collections
import colorsys
import itertools
import os

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
    plt.savefig(plot_path)
    plt.clf()


def create_distribution_plots(agg_results: list, output_dir: str):
    scores_by_dataset = agg_results[0]['test_scores']['scores_by_dataset_id']
    if len(scores_by_dataset) == 0:
        print('No results, skipping')
        return

    scores_by_dataset_keys = scores_by_dataset.keys()
    metric_keys = scores_by_dataset[list(scores_by_dataset_keys)[0]].keys()
    results_dict = {}
    # aggregates metrics over all datasets
    for metric_key in metric_keys:
        new_metric_list = []
        for index, dataset_name in enumerate(scores_by_dataset_keys):
            new_metric_list.append(scores_by_dataset[dataset_name][metric_key])
        # non-iterable metrics are only non-iterable two layers deep now
        if isinstance(new_metric_list[0][0], collections.Iterable):
            new_metric_list = list(zip(*new_metric_list)) # now we have tuples of tuples for each metric
            flattened_results = [list(sum(tupleOfTuples, ())) for tupleOfTuples in new_metric_list]
        else:
            flattened_results = list(itertools.chain(*new_metric_list)) # flatten all the arrays into one array
        results_dict[metric_key] = flattened_results

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # plot aggregate scores in violin plot and save to `output_dir`
    for metric_key in metric_keys:
        if isinstance(results_dict[metric_key][0], collections.Iterable):
            for index in [1, 25, 100, -1]:
                name_of_index = 'all' if index == -1 else str(index)
                name_of_metric = metric_key.replace('_', ' ')
                ax = sns.violinplot(x=results_dict[metric_key][index], cut=0)
                plt.title('Distribution of {} at K={}'.format(name_of_metric, name_of_index))
                plt.xlabel('{} at K={}'.format(name_of_metric, name_of_index))
                plt.ylabel('Frequency')
                plt.savefig(os.path.join(output_dir, '{}-at-{}-violin-plot.png'.format(metric_key, name_of_index)))
                plt.close()
        else:
            ax = sns.violinplot(x=results_dict[metric_key], cut=0)
            plt.title('Distribution of Metric: {}'.format(metric_key))
            plt.xlabel('{}'.format(metric_key))
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(output_dir, '{}-violin-plot.png'.format(metric_key)))
            plt.close()
