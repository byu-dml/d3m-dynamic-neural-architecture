import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

from .kND import KNearestDatasets


class AutoSklearnMetalearner():

    def __init__(self, metric='test_accuracy', maximize_metric=True):
        # if metadata_path is None:
        self.runs = None
        self.test_runs = None
        self.metafeatures = None
        self.datasets = []
        self.testset = []
        self.pipeline_descriptions = {}
        self.metric = metric
        self.maximize_metric = maximize_metric
        if self.maximize_metric:
            self.opt = np.nanmax
        else:
            self.opt = np.nanmin

    def process_metadata(self, data_type="train", validation_names=None):
        """
        Reads in a dataset from a static json file.  Loads lots of the information needed to process the KnD
        :param data_type: "train" or "test"
        :param validation_names:
        :return:
        """
        dataset_to_use = self.datasets if data_type == "train" else self.testset
        runs_by_pipeline = {}
        metafeatures = {}
        test_metafeatures = {}
        for row in self.metadata:
            dataset_name = row['dataset']
            pipeline_id = row['pipeline_id']
            run = row[self.metric]
            if dataset_name not in metafeatures:
                dataset_to_use.append(dataset_name)
                metafeatures[dataset_name] = row['metafeatures']
            if pipeline_id not in self.pipeline_descriptions:
                self.pipeline_descriptions[pipeline_id] = row['pipeline']
            if dataset_name not in runs_by_pipeline:
                runs_by_pipeline[dataset_name] = {}
            runs_by_pipeline[dataset_name][pipeline_id] = run
        default_runs = {pipeline_id: np.nan for pipeline_id in self.pipeline_descriptions}
        for dataset_name in dataset_to_use:
            runs = runs_by_pipeline[dataset_name]
            runs_by_pipeline[dataset_name] = copy.deepcopy(default_runs)
            runs_by_pipeline[dataset_name].update(runs)
        if data_type == "train":
            self.runs = pd.DataFrame(runs_by_pipeline)
            if validation_names is not None:
                # we are using a validation set, which is passed along
                self.test_runs = self.runs.copy(deep=True)
                self.test_runs = self.test_runs[validation_names]
                self.runs = self.runs.drop(validation_names, axis=1, inplace=False)
                # self.test_runs.columns = np.arange(len(validation_names))
                # self.runs.columns = np.arange(len(dataset_to_use) - len(validation_names))
                self.datasets = list(set(dataset_to_use).difference(set(validation_names)))
                self.testset = validation_names
            else:
                self.runs.columns = np.arange(len(dataset_to_use))
        else:
            self.test_runs = pd.DataFrame(runs_by_pipeline)
            self.test_runs.columns = np.arange(len(dataset_to_use))
        self.metafeatures = pd.DataFrame(metafeatures)
        if validation_names is not None:
            self.test_metafeatures = self.metafeatures.copy(deep=True)
            self.test_metafeatures = self.test_metafeatures[validation_names]
            self.metafeatures = self.metafeatures.drop(validation_names, axis=1, inplace=False)
            # self.metafeatures.columns = np.arange(self.metafeatures.shape[1])
            # self.test_metafeatures.columns = np.arange(self.test_metafeatures.shape[1])


    def get_k_best_pipelines(self, dataset_metafeatures, all_other_metafeatures, runs, k, current_dataset_name):
        mf_mask = np.arange(dataset_metafeatures.shape[0])[np.isfinite(dataset_metafeatures)]
        dataset_metafeatures = dataset_metafeatures.iloc[mf_mask]
        # all_other_metafeatures = all_other_metafeatures.iloc[:, mf_mask]
        all_other_metafeatures = all_other_metafeatures.replace([np.inf, -np.inf], np.nan)
        all_other_metafeatures = all_other_metafeatures.fillna(all_other_metafeatures.mean(skipna=True))
        all_other_metafeatures = all_other_metafeatures.transpose()

        # get the ids for pipelines that we have real values for
        current_validation_ids = self.validation_set.loc[self.validation_set["index"] == current_dataset_name]["pipeline_ids"].iloc[0]

        kND = KNearestDatasets(metric='l1', random_state=3)
        kND.fit(all_other_metafeatures, runs, current_validation_ids, self.maximize_metric)
        # best suggestions is a list of 3-tuples that contain the pipeline index, the distance value, and the pipeline_id

        best_suggestions = kND.kBestSuggestions(dataset_metafeatures, k=k)
        k_best_pipelines = [suggestion[2] for suggestion in best_suggestions]
        return k_best_pipelines

    def get_k_best_pipelines_per_dataset(self, k):
        k_best_pipelines_per_dataset = {}
        for dataset_index, dataset_name in enumerate(self.testset):
            # each column is a metafeature for the column which is a dataset name
            dataset_metafeatures = self.test_metafeatures.iloc[:, dataset_index]
            # we want to drop the metafeatures from the group if we are given a dataset.  Otherwise we have seperate metafeatures
            all_other_metafeatures = self.metafeatures.drop(index=dataset_index) if not len(self.test_metafeatures) else self.metafeatures
            runs = self.runs.drop(columns=dataset_index) if not len(self.test_metafeatures) else self.runs
            pipelines = self.get_k_best_pipelines(dataset_metafeatures, all_other_metafeatures, runs, k, dataset_name)
            k_best_pipelines_per_dataset[dataset_name] = pipelines
        return k_best_pipelines_per_dataset

    def get_metric_difference_from_best(self, k):
        metric_differences = {}
        top_pipeline_performance = []
        top_k_out_of_total = []
        top_pipelines_per_dataset = {}
        k_best_pipelines_per_dataset = self.get_k_best_pipelines_per_dataset(k)
        for dataset_index, dataset_name in enumerate(self.testset):
            k_best_pipelines = k_best_pipelines_per_dataset[dataset_name]
            metric_value = self.opt([self.runs.loc[pipeline].iloc[dataset_index] for pipeline in k_best_pipelines if np.isfinite(self.runs.loc[pipeline].iloc[dataset_index])])
            best_metric_value = self.opt(self.test_runs.iloc[:, dataset_index])
            # gather the actual top pipelines for each test dataset
            top_pipelines_per_dataset[dataset_name] = list(self.test_runs.iloc[:, dataset_index].nlargest(k).index)
            # get the number of top k pipelines that are actually in the top k
            top_ids_actual = top_pipelines_per_dataset[dataset_name]
            top_k_out_of_total.append(len(set(top_ids_actual).intersection(set(k_best_pipelines))))
            # get the actual values for predicted top pipeline
            top_pipeline_performance.append(best_metric_value)
            metric_differences[dataset_name] = np.abs(best_metric_value - metric_value)

        print("The top k of k is, on average:", np.mean(top_k_out_of_total))
        print("The difference in predicted metric vs actual is", np.mean(list(metric_differences.values())))
        return metric_differences, top_pipeline_performance, top_k_out_of_total, top_pipelines_per_dataset


    def predict(self, k, validation_set: dict):
        """
        A wrapper for all the other functions so that this is organized
        :param k: number of datasets
        :param validation_set: a dictionary containing pipelines, ids, and real f1 scores. MUST CONTAIN PIPELINE IDS
        from each dataset being passed in.  This is used for the rankings
        :return:
        """
        self.validation_set = pd.DataFrame(validation_set).transpose()
        # make indexes columns
        self.validation_set.reset_index(level=0, inplace=True)
        validate_dataset_names = list(validation_set.keys())
        return self.get_metric_difference_from_best(k)

    def fit(self, use_static_test=False, validation_names=None):
        """
        A basic KNN fit.  Loads in and processes the training data from a fixed split to avoid working with the dataloader
        :param use_static_test:
        :return:
        """
        if use_static_test:
            # these are in this order so the metadata holds the train and self.datasets and self.testsets get filled
            with open(os.path.join(os.getcwd(), "dna/data", "test_data.json"), 'r') as f:
                self.metadata = json.load(f)
            self.process_metadata(data_type="test")
            with open(os.path.join(os.getcwd(), "dna/data", "train_data.json"), 'r') as f:
                self.metadata = json.load(f)
            self.process_metadata(data_type="train")
        else:
            with open(os.path.join(os.getcwd(), "dna/data", "train_data.json"), 'r') as f:
                self.metadata = json.load(f)
                self.process_metadata(data_type="train", validation_names=validation_names)


"""
This may or may not work..  It is not support currently from the command line
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=25)
    parser.add_argument('--regression', action='store_true')

    args = parser.parse_args()

    if args.regression:
        # the metric stays test accuracy because this is the name of the metric in our metadata file, however if that is fixed this should change
        metric = 'test_accuracy'
        maximize_metric = False
    else:
        metric = 'test_accuracy'
        maximize_metric = True

    metalearner = AutoSklearnMetalearner(args.metadata, args.seed, metric, maximize_metric)
    metric_differences = metalearner.get_metric_difference_from_best(args.k)
    mean_difference = np.mean(list(metric_differences.values()))
    print(mean_difference)

