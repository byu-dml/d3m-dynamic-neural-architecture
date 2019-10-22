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


def plot_violin_of_score_distributions(distributions_by_model_num, model_mapping, score_name, plot_path):
    model_counter = []
    values = []
    for model_num in distributions_by_model_num.keys():
        values.extend(distributions_by_model_num[model_num])
        model_counter.extend([model_mapping[model_num] for i in range(len(distributions_by_model_num[model_num]))])
    plot_df = pd.DataFrame({"model_number": model_counter, "values": values})
    # make it prettier
    ax = sns.violinplot(x="model_number", y="values", data=plot_df, cut=0)
    plt.title('Distribution of Metric: {}'.format(score_name))
    plt.xlabel('{}'.format(score_name))
    plt.ylabel('Frequency')
    plt.savefig(plot_path)
    plt.close()


def create_distribution_plots(results: pd.DataFrame, output_dir: str, list_of_k: list):
    model_mapping = results["model_id"].tolist() # index maps id to model name
    # gather relevant column names
    use_cols = [col_name for col_name in results.columns if 'scores_by_dataset' in col_name]
    results = results[use_cols]
    # gather relevant metrics
    metrics = set([col_name.split(".")[-1] for col_name in use_cols])

    distribution_dict = {}
    # go through metric by metric
    for metric in metrics:
        # find the relevant column
        for col_name in results:
            for model_num in range(len(results[col_name])):
                model_data = results[col_name][model_num]
                if metric in col_name:

                    # initialize the metric dict
                    if metric not in distribution_dict:
                        if "at_k" in col_name:
                            for k in list_of_k:
                                new_metric_name = metric + "_k={}".format(k if k != -1 else "all")
                                # only make a new dict if it's not present
                                if new_metric_name not in distribution_dict:
                                    distribution_dict[new_metric_name] = {}
                        else:   
                            distribution_dict[metric] = {}

                    # aggregate
                    if "at_k" not in col_name:
                        if metric in distribution_dict and model_num in distribution_dict[metric]:
                            distribution_dict[metric][model_num].extend(model_data)
                        else:
                            # initialize the dict
                            distribution = model_data # each one is a series of a list
                            distribution_dict[metric][model_num] = distribution
                    else:
                        # we have a run by K series-list structure -> turn into n by K array, zipping by the longest and filling with nans
                        pad = len(max(model_data, key=len))
                        distribution_array = np.array([i + [0]*(pad-len(i)) for i in model_data])
                        for k in list_of_k:
                            if distribution_array.shape[1] < k:
                                # if there are less than k pipelines, skip it
                                continue
                            new_metric_name = metric + "_k={}".format(k if k != -1 else "all")
                            distribution = distribution_array[:, k] # get the k-th columns, which is the distribution at that k
                            if metric in distribution_dict and model_num in distribution_dict[metric]:
                                distribution_dict[new_metric_name][model_num].extend(distribution.tolist())
                            else:
                                # initialize the dict mapping to a list so that we can extend it
                                distribution_dict[new_metric_name][model_num] = distribution.tolist() 
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # plot aggregate scores in violin plot and save to `output_dir`
    for metric_key in distribution_dict.keys():
        score_name = metric_key.replace("_k_by_run", "")
        plot_violin_of_score_distributions(
            distribution_dict[metric_key], model_mapping, score_name=score_name,
            plot_path=os.path.join(output_dir, '{}-violin-plot.png'.format(score_name))
        )
