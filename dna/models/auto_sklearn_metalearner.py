import numpy as np
import pandas as pd
import os
import json

from .base_models import RankModelBase
from .kND import KNearestDatasets


class AutoSklearnMetalearner(RankModelBase):

    def __init__(self, seed=0):
        RankModelBase.__init__(self, seed=seed)

    def get_k_best_pipelines(self, data, dataset_metafeatures, all_other_metafeatures, runs, current_dataset_name):
        # all_other_metafeatures = all_other_metafeatures.iloc[:, mf_mask]
        all_other_metafeatures = all_other_metafeatures.replace([np.inf, -np.inf], np.nan)
        # this should aready be done by the time it gets here
        all_other_metafeatures = all_other_metafeatures.transpose()
        # get the metafeatures out of their list
        all_other_metafeatures = pd.DataFrame(all_other_metafeatures.iloc[1].tolist(), index=all_other_metafeatures.iloc[0])
        all_other_metafeatures = all_other_metafeatures.fillna(all_other_metafeatures.mean(skipna=True))
        all_other_metafeatures = all_other_metafeatures.reset_index().drop_duplicates()
        all_other_metafeatures = all_other_metafeatures.set_index('dataset_id')
        # get the ids for pipelines that we have real values for
        current_validation_ids = set(pipeline['id'] for pipeline in data.pipeline)

        kND = KNearestDatasets(metric='l1', random_state=3)
        kND.fit(all_other_metafeatures, self.run_lookup, current_validation_ids, self.maximize_metric)
        # best suggestions is a list of 3-tuples that contain the pipeline index,the distance value, and the pipeline_id
        best_suggestions = kND.kBestSuggestions(pd.Series(dataset_metafeatures), k=all_other_metafeatures.shape[0])
        k_best_pipelines = [suggestion[2] for suggestion in best_suggestions]
        return k_best_pipelines

    def get_k_best_pipelines_per_dataset(self, data):
        # they all should have the same dataset and metafeatures so take it from the first row
        dataset_metafeatures = data["metafeatures"].iloc[0]
        dataset_name = data["dataset_id"].iloc[0]
        all_other_metafeatures = self.metafeatures
        pipelines = self.get_k_best_pipelines(data, dataset_metafeatures, all_other_metafeatures, self.runs, dataset_name)
        return pipelines


    def predict_rank(self, data, *, verbose=False):
        """
        A wrapper for all the other functions so that this is organized
        :data: a dictionary containing pipelines, ids, and real f1 scores. MUST CONTAIN PIPELINE IDS
        from each dataset being passed in.  This is used for the rankings
        :return:
        """
        data = pd.DataFrame(data)
        k_best_pipelines_per_dataset = self.get_k_best_pipelines_per_dataset(data)
        return {
            'pipeline_id': k_best_pipelines_per_dataset,
            'rank': list(range(len(k_best_pipelines_per_dataset))),
        }

    def fit(self, training_dataset=None, metric='test_accuracy', maximize_metric=True, *, validation_data=None, output_dir=None, verbose=False):
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
            self.metafeatures = pd.DataFrame(self.metadata)[['dataset_id', 'metafeatures']]
            self.runs = pd.DataFrame(self.metadata)[['dataset_id', 'pipeline', 'test_f1_macro']]
            self.run_lookup = self.process_runs()

    def process_runs(self):
        """
        This function is used to transform the dataframe into a workable object fot the KNN, with rows of pipeline_ids
        and columns of datasets, with the inside being filled with the scores
        :return:
        """
        new_runs = {}
        for index, row in self.runs.iterrows():
            dataset_name = row["dataset_id"]
            if dataset_name not in new_runs:
                new_runs[dataset_name] = {}
            else:
                new_runs[dataset_name][row["pipeline"]['id']] = row['test_f1_macro']
        final_new = pd.DataFrame(new_runs)
        return final_new
