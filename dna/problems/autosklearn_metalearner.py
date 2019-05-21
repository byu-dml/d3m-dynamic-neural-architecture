import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

from .kND import KNearestDatasets


class AutoSklearnMetalearner():

    def __init__(self, ):
        pass


    def get_k_best_pipelines(self, dataset_metafeatures, all_other_metafeatures, runs, k, current_dataset_name):
        # all_other_metafeatures = all_other_metafeatures.iloc[:, mf_mask]
        all_other_metafeatures = all_other_metafeatures.replace([np.inf, -np.inf], np.nan)
        # this should aready be done by the time it gets here
        all_other_metafeatures = all_other_metafeatures.transpose()
        # get the metafeatures out of their list
        all_other_metafeatures = pd.DataFrame(all_other_metafeatures.iloc[1].tolist(), index=all_other_metafeatures.iloc[0])
        all_other_metafeatures = all_other_metafeatures.fillna(all_other_metafeatures.mean(skipna=True))
        all_other_metafeatures = all_other_metafeatures.reset_index().drop_duplicates()
        all_other_metafeatures = all_other_metafeatures.set_index('dataset')
        # get the ids for pipelines that we have real values for
        current_validation_ids = self.validation_set.pipeline_id.unique()

        kND = KNearestDatasets(metric='l1', random_state=3)
        kND.fit(all_other_metafeatures, self.run_lookup, current_validation_ids, self.maximize_metric)
        # best suggestions is a list of 3-tuples that contain the pipeline index,the distance value, and the pipeline_id
        best_suggestions = kND.kBestSuggestions(pd.Series(dataset_metafeatures), k=k)
        k_best_pipelines = [suggestion[2] for suggestion in best_suggestions]
        return k_best_pipelines

    def get_k_best_pipelines_per_dataset(self, k):
        # they all should have the same dataset and metafeatures so take it from the first row
        dataset_metafeatures = self.validation_set["metafeatures"].iloc[0]
        dataset_name = self.validation_set["dataset"].iloc[0]
        all_other_metafeatures = self.metafeatures
        pipelines = self.get_k_best_pipelines(dataset_metafeatures, all_other_metafeatures, self.runs, k, dataset_name)
        return pipelines


    def predict_rank(self, metadata_for_one_dataset, k, dataset_name):
        """
        A wrapper for all the other functions so that this is organized
        :param k: number of datasets
        :param validation_set: a dictionary containing pipelines, ids, and real f1 scores. MUST CONTAIN PIPELINE IDS
        from each dataset being passed in.  This is used for the rankings
        :return:
        """
        self.validation_set = metadata_for_one_dataset
        k_best_pipelines_per_dataset = self.get_k_best_pipelines_per_dataset(k)
        import pdb; pdb.set_trace()
        ranked_df = self.run_lookup.loc[k_best_pipelines_per_dataset][dataset_name]
        ranked_df.reset_index(inplace=True)
        ranked_df.columns = ["id", "score"]

        actual_df = self.test_runs.iloc[:, 0].to_frame("score")
        actual_df.reset_index(inplace=True)
        actual_df.columns = ["id", "score"]

        return ranked_df, actual_df

    def fit(self, training_dataset=None, metric='test_accuracy', maximize_metric=True):
        """
        A basic KNN fit.  Loads in and processes the training data from a fixed split
        :param training_dataset: the dataset to be processed.  If none given it will be pulled from the hardcoded file
        :param metric: what kind of metric we're using in our metalearning
        :param maximize_metric: whether to maximize or minimize that metric.  Defaults to Maximize
        """
        # if metadata_path is None:
        self.runs = None
        self.test_runs = None
        self.metafeatures = None
        self.datasets = []
        self.testset = []
        self.pipeline_descriptions = {}
        self.metric = metric
        self.maximize_metric = maximize_metric
        self.opt = np.nanmax
        if training_dataset is None:
            # these are in this order so the metadata holds the train and self.datasets and self.testsets get filled
            with open(os.path.join(os.getcwd(), "dna/data", "test_data.json"), 'r') as f:
                self.metadata = json.load(f)
            self.process_metadata(data_type="test")
            with open(os.path.join(os.getcwd(), "dna/data", "train_data.json"), 'r') as f:
                self.metadata = json.load(f)
            self.process_metadata(data_type="train")
        else:
            self.metadata = training_dataset
            self.metafeatures = pd.DataFrame(self.metadata)[['dataset', 'metafeatures']]
            self.runs = pd.DataFrame(self.metadata)[['dataset', 'pipeline_id', 'test_accuracy']]
            self.run_lookup = self.process_runs()

    def process_runs(self):
        """
        This function is used to transform the dataframe into a workable object fot the KNN, with rows of pipeline_ids
        and columns of datasets, with the inside being filled with the scores
        :return:
        """
        new_runs = {}
        for index, row in self.runs.iterrows():
            dataset_name = row["dataset"]
            if dataset_name not in new_runs:
                new_runs[dataset_name] = {}
            else:
                new_runs[dataset_name][row["pipeline_id"]] = row["test_accuracy"]
        final_new = pd.DataFrame(new_runs)
        return final_new

    """
    Saving this in case we want to use it to pull the static test set
    No good reason to keep it otherwise
    """
    def process_metadata(self, data_type="train", validation_names=None):
        """
        Creates the dataframes needed and stores them
        """
        dataset_to_use = self.datasets if data_type == "train" else self.testset
        runs_by_pipeline = {}
        metafeatures = {}
        test_metafeatures = {}
        import pdb; pdb.set_trace()
        for row in dataset_to_use:
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
            self.metafeatures.columns = np.arange(self.metafeatures.shape[1])
            self.test_metafeatures.columns = np.arange(self.test_metafeatures.shape[1])




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

