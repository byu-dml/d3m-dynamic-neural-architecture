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


def create_distribution_plots(regression_results: pd.DataFrame, rank_results: pd.DataFrame, output_dir: str):
    # gather relevant column names
    reg_cols = [col_name for col_name in regression_results.columns if 'scores_by_dataset' in col_name]
    rank_cols = [col_name for col_name in rank_results.columns if 'scores_by_dataset' in col_name]
    regression_results = regression_results[reg_cols]
    rank_results = rank_results[rank_cols]

    # gather relevant metrics
    reg_metrics = set([col_name.split(".")[-1] for col_name in reg_cols])
    rank_metrics = set([col_name.split(".")[-1] for col_name in rank_cols])


    distribution_dict = {}
    # go through both problems
    for problem_name, problem_metrics, problem_dataset in [("regression", reg_metrics, regression_results), ("rank", rank_metrics, rank_results)]:
        # go through metric by metric
        for metric in problem_metrics:
            # find the relevant column in the problem results
            for col_name in problem_dataset:
                assert len(problem_dataset[col_name]) == 1, "had more than one list in series, error"
                if metric in col_name:
                    if "at_k" not in col_name:
                        if metric in distribution_dict:
                            distribution_dict[metric].extend(problem_dataset[col_name][0])
                        else:
                            # initialize the dict
                            distribution = problem_dataset[col_name][0] # each one is a series of a list
                            distribution_dict[metric] = distribution
                    else:
                        # we have a run by K series-list structure -> turn into n by K array, zipping by the longest and filling with nans
                        pad = len(max(problem_dataset[col_name][0], key=len))
                        distribution_array = np.array([i + [0]*(pad-len(i)) for i in problem_dataset[col_name][0]])
                        for k in [1, 25, 100, -1]:
                            if distribution_array.shape[1] < k:
                                # if there are less than k pipelines, skip it
                                continue
                            new_metric_name = metric + "_k={}".format(k if k != -1 else "all")
                            distribution = distribution_array[:, k] # get the k-th columns, which is the distribution at that k
                            if metric in distribution_dict:
                                distribution_dict[new_metric_name].extend(distribution.tolist())
                            else:
                                # initialize the dict mapping to a list so that we can extend it
                                distribution_dict[new_metric_name] = distribution.tolist() 
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # plot aggregate scores in violin plot and save to `output_dir`
    for metric_key in distribution_dict.keys():
        # make it prettier
        metric_key_name = metric_key.replace("_k_by_run", "")
        ax = sns.violinplot(x=distribution_dict[metric_key], cut=0)
        plt.title('Distribution of Metric: {}'.format(metric_key_name))
        plt.xlabel('{}'.format(metric_key_name))
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, '{}-violin-plot.png'.format(metric_key_name)))
        plt.close()
