# Code taken from https://github.com/automl/auto-sklearn/blob/master/autosklearn/metalearning/metalearning/kNearestDatasets/kND.py

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
import sklearn.utils


class KNearestDatasets(object):
    def __init__(self, metric='l1', random_state=None, metric_params=None, rank_distance_metric=None):

        self.metric = metric
        self.rank_distance_metric = rank_distance_metric
        self.model = None
        self.metric_params = metric_params
        self.metafeatures = None
        self.runs = None
        self.best_configuration_per_dataset = None
        self.random_state = sklearn.utils.check_random_state(random_state)

        if self.metric_params is None:
            self.metric_params = {}

    def fit(self, metafeatures, pipeline_runs, validation_set_pipelines, maximize_metric=True):
        """Fit the Nearest Neighbor model.
        Parameters
        ----------
        metafeatures : pandas.DataFrame
            A pandas dataframe. Each row represents a dataset, each column a
            metafeature.
        runs : dict
            A pandas dataframe containing a list of runs for each dataset.
        """
        assert isinstance(metafeatures, pd.DataFrame)
        # assert metafeatures.values.dtype in (np.float32, np.float64)
        # assert np.isfinite(metafeatures.values).all()
        assert isinstance(pipeline_runs, pd.DataFrame)
        # this matching column should be dataset number
        assert pipeline_runs.shape[1] == metafeatures.shape[0], \
            (pipeline_runs.shape[1], metafeatures.shape[0])


        """
        Need for runs: pipeline X dataset
        Need for metafeatures: dataset X metafeatures 
        """

        self.metafeatures = metafeatures
        runs = pipeline_runs.copy(deep=True)
        self.runs = runs.copy(deep=True)
        self.num_datasets = runs.shape[1]

        # for each dataset, sort the runs according to their result
        best_configuration_per_dataset = {}
        all_configuration_per_dataset = {}

        if maximize_metric:
            opt = np.nanargmax
        else:
            opt = np.nanargmin
        for dataset_name in runs:
            if not np.isfinite(runs[dataset_name]).any():
                best_configuration_per_dataset[dataset_name] = None
                all_configuration_per_dataset[dataset_name] = None
            else:
                configuration_idx = ""
                # TODO: I added this.  Should I take it out?
                while configuration_idx not in validation_set_pipelines:
                    opt_index = opt(runs[dataset_name].values)
                    configuration_idx = runs[dataset_name].index[opt_index]
                    runs[dataset_name].iloc[opt_index] = np.nan

                best_configuration_per_dataset[dataset_name] = configuration_idx
                all_configuration_per_dataset[dataset_name] = runs[dataset_name].reset_index()

        self.best_configuration_per_dataset = best_configuration_per_dataset
        self.all_configuration_per_dataset = all_configuration_per_dataset

        if callable(self.metric):
            self._metric = self.metric
            self._p = 0
        elif self.metric.lower() == "l1":
            self._metric = "minkowski"
            self._p = 1
        elif self.metric.lower() == "l2":
            self._metric = "minkowski"
            self._p = 2
        else:
            raise ValueError(self.metric)

        self._nearest_neighbors = NearestNeighbors(
            n_neighbors=self.num_datasets, radius=None, algorithm="brute",
            leaf_size=30, metric=self._metric, p=self._p,
            metric_params=self.metric_params)

    def kNearestDatasets(self, x, k=1, return_distance=False):
        """Return the k most similar datasets with respect to self.metric
        Parameters
        ----------
        x : pandas.Series
            A pandas Series object with the metafeatures for one dataset
        k : int
            Number of k nearest datasets which are returned. If k == -1,
            return all dataset sorted by similarity.
        return_distance : bool, optional. Defaults to False
            If true, distances to the new dataset will be returned.
        Returns
        -------
        list
            Names of the most similar datasets, sorted by similarity
        list
            Sorted distances. Only returned if return_distances is set to True.
        """
        assert type(x) == pd.Series
        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        elif k == -1:
            k = self.num_datasets
        # no need to scale, it was done earlier
        # X_train, x = self._scale(self.metafeatures, x)
        X_train = self.metafeatures
        x = x.values.reshape((1, -1))
        self._nearest_neighbors.fit(X_train)
        distances, neighbor_indices = self._nearest_neighbors.kneighbors(
            x, n_neighbors=k, return_distance=True)

        assert k == neighbor_indices.shape[1]

        rval = [self.metafeatures.index[i]
                # Neighbor indices is 2d, each row are the indices for one
                # dataset in x.
                for i in neighbor_indices[0]]

        if return_distance is False:
            return rval
        else:
            return rval, distances[0]

    def kBestSuggestions(self, x, k=1, exclude_double_configurations=True):
        assert type(x) == pd.Series

        if k < -1 or k == 0:
            raise ValueError('Number of neighbors k cannot be zero or negative.')
        nearest_datasets, distances = self.kNearestDatasets(x, -1, return_distance=True)
        kbest = []

        added_configurations = set()
        # get the top 25 best pipelines from each dataset
        for dataset_name, distance in zip(nearest_datasets, distances):
            best_configuration = self.best_configuration_per_dataset[dataset_name]

            if best_configuration is None:
                continue

            if exclude_double_configurations:
                # make sure the best pipelines we're returning have actual values and are not already part of our top 25
                if best_configuration not in added_configurations:
                    added_configurations.add(best_configuration)
                    kbest.append((dataset_name, distance, best_configuration))
            else:
                kbest.append((dataset_name, distance, best_configuration))

            if k != -1 and len(kbest) >= k:
                break

        if k == -1:
            k = len(kbest)
        return kbest[:k]

    def allBestSuggestions(self, x, exclude_double_configurations=True):
        """
        This is our implementation of nearest neighbors to grab the ranking of pipelines.
        """
        assert type(x) == pd.Series

        
        nearest_datasets, distances = self.kNearestDatasets(x, -1, return_distance=True)
        kbest = []

        added_configurations = set()
        initialized = False
        # add the distance ranking to each dataset scores
        for dataset_name, distance in zip(nearest_datasets, distances):
            all_configurations = self.all_configuration_per_dataset[dataset_name]
            if initialized == False:
                # create the empty dataframe
                all_pipelines_ranked = pd.DataFrame(columns=["pipeline", "rank"])
                initialized = True

            # weight the scores by the distance
            all_configurations[dataset_name] = self.rank_distance_metric(all_configurations[dataset_name], distance)
            all_configurations.columns = all_pipelines_ranked.columns
            all_pipelines_ranked = all_pipelines_ranked.append(all_configurations, ignore_index=True)

        # pick the top ranked pipeline for each pipeline
        row_list = []
        for pipeline in all_pipelines_ranked["pipeline"].unique():
            rank_score = all_pipelines_ranked[all_pipelines_ranked["pipeline"] == pipeline]["rank"].max()
            row_list.append({"pipeline": pipeline, "rank": rank_score})

        final_df = pd.DataFrame(row_list)
        final_df.sort_values("rank", inplace=True)
        return final_df["pipeline"].tolist()

    """
    Scaling should have already been done in the preprocessing part of DNA 
    """
    def _scale(self, metafeatures, other):
        assert isinstance(other, pd.Series), type(other)
        assert other.values.dtype == np.float64
        scaled_metafeatures = metafeatures.copy(deep=True)
        other = other.copy(deep=True)

        mins = scaled_metafeatures.min()
        maxs = scaled_metafeatures.max()
        # I also need to scale the target datasetself.mself. meta features...
        mins = pd.DataFrame(data=[mins, other]).min()
        maxs = pd.DataFrame(data=[maxs, other]).max()
        divisor = (maxs-mins)
        divisor[divisor == 0] = 1
        scaled_metafeatures = (scaled_metafeatures - mins) / divisor
        other = (other - mins) / divisor
        scaled_metafeatures = scaled_metafeatures.fillna(0)
        other = other.fillna(0)
        return scaled_metafeatures, other
