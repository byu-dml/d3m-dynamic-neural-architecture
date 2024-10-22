import os
import random
import shutil

import autosklearn.regression as autosklearn
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import torch
from .torch_modules.mlp import MLP

from .base_models import RankModelBase, RegressionModelBase, SklearnBase, PyTorchRegressionRankModelBase
from dna import utils
from dna.kND import KNearestDatasets


class MeanBaseline(RegressionModelBase):

    # used for plotting and reporting
    name = 'Mean'
    color = 'black'

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.mean = None

    def fit(self, data, *, output_dir=None, verbose=False):
        total = 0
        for instance in data:
            total += instance['test_f1_macro']
        self.mean = total / len(data)
        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.mean is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.mean] * len(data)


class MedianBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.median = None

    def fit(self, data, *, output_dir=None, verbose=False):
        self.median = np.median([instance['test_f1_macro'] for instance in data])
        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.median is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.median] * len(data)


class PerPrimitiveBaseline(RegressionModelBase, RankModelBase):

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.primitive_scores = None

    def fit(self, data, *, output_dir=None, verbose=False):
        # for each primitive, get the scores of all the pipelines that use the primitive
        primitive_score_totals = {}
        for instance in data:
            for primitive in instance['pipeline']['steps']:
                if primitive['name'] not in primitive_score_totals:
                    primitive_score_totals[primitive['name']] = {
                        'total': 0,
                        'count': 0,
                    }
                primitive_score_totals[primitive['name']]['total'] += instance['test_f1_macro']
                primitive_score_totals[primitive['name']]['count'] += 1

        # compute the average pipeline score per primitive
        self.primitive_scores = {}
        for primitive_name in primitive_score_totals:
            total = primitive_score_totals[primitive_name]['total']
            count = primitive_score_totals[primitive_name]['count']
            self.primitive_scores[primitive_name] = total / count

        self.fitted = True

    def predict_regression(self, data, **kwargs):
        if self.primitive_scores is None:
            raise ModelNotFitError('PerPrimitiveBaseline not fit')

        predictions = []
        for instance in data:
            prediction = 0
            for primitive in instance['pipeline']['steps']:
                prediction += self.primitive_scores[primitive['name']]
            prediction /= len(instance['pipeline']['steps'])
            predictions.append(prediction)

        return predictions

    def predict_rank(self, data, **kwargs):
        predictions = self.predict_regression(data, **kwargs)
        ranks = list(utils.rank(predictions))
        return {
            'pipeline_id': [instance['pipeline_id'] for instance in data],
            'rank': ranks,
        }


class RandomBaseline(RankModelBase):

    # used for plotting and reporting
    name = 'Random'
    color = 'silver'

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self._random_state = np.random.RandomState(seed)
        self.fitted = True

    def fit(self, *args, **kwargs):
        pass

    def predict_rank(self, data, *, verbose=False):
        predictions = list(range(len(data)))
        self._random_state.shuffle(predictions)
        return {
            'pipeline_id': [instance['pipeline_id'] for instance in data],
            'rank': predictions,
        }


class LinearRegressionBaseline(SklearnBase):

    # used for plotting and reporting
    name = 'Linear Regression'
    color = 'saddlebrown'

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.regressor = linear_model.LinearRegression()
        self.fitted = False


class RandomForestBaseline(SklearnBase):

    # used for plotting and reporting
    name = 'Random Forest'
    color = 'forestgreen'

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.regressor = RandomForestRegressor(random_state=seed)
        self.fitted = False


class MetaAutoSklearn(SklearnBase):

    # used for plotting and reporting
    name = 'Meta Auto-sklearn'
    color = 'blue'

    def __init__(self, seed=0, **kwargs):
        super().__init__(seed=seed)

        tmp_dir = self._init_tmp_dir()
        while os.path.isdir(tmp_dir):
            tmp_dir = self._init_tmp_dir()

        self.regressor = autosklearn.AutoSklearnRegressor(seed=seed, **kwargs, tmp_folder=tmp_dir)
        self.fitted = False

    @staticmethod
    def _init_tmp_dir():
        return os.path.join('.', 'tmp', '{}'.format(random.randint(2**63, 2**64-1)))


class AutoSklearnMetalearner(RegressionModelBase, RankModelBase):

    # used for plotting and reporting
    name = 'k-ND (Auto-sklearn)'
    color = 'gold'

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self._knd = KNearestDatasets(metric='l1')

    def _predict(self, data, method, k=None):
        data = pd.DataFrame(data)
        # they all should have the same dataset and metafeatures so take it from the first row
        dataset_metafeatures = pd.Series(data['metafeatures'].iloc[0])
        queried_pipelines = data['pipeline_id']

        if method == 'all':
            predicted_pipelines = self._knd.knn_regression(dataset_metafeatures)
            predicted_pipelines = predicted_pipelines.sort_values(ascending=False).index.tolist()
        elif method == 'k':
            predicted_pipelines = self._knd.kBestSuggestions(dataset_metafeatures, k=k)
        else:
            raise ValueError('Unknown method: {}'.format(method))

        for pipeline_id in set(predicted_pipelines).difference(set(queried_pipelines)):
            predicted_pipelines.remove(pipeline_id)

        return predicted_pipelines

    def predict_regression(self, data, **kwargs):
        predictions = []
        cached_predictions = {}
        for instance in data:
            dataset_id = instance['dataset_id']
            if dataset_id not in cached_predictions:
                metafeatures = pd.Series(instance['metafeatures'])
                cached_predictions[dataset_id] = self._knd.knn_regression(metafeatures)
            pipeline_id = instance['pipeline_id']
            predictions.append(cached_predictions[dataset_id].get(pipeline_id, None))
        return predictions

    def predict_rank(self, data, **kwargs):
        """
        Attempts to rank all pipelines in data, but can only rank pipelines found in the training data.
        """
        ranked_pipelines = self._predict(data, method='k', k=len(data))

        return {
            'pipeline_id': ranked_pipelines,
            'rank': list(range(len(ranked_pipelines))),
        }

    def fit(self, train_data, *args, **kwargs):
        self._runs = self._process_runs(train_data)
        self._metafeatures = self._process_metafeatures(train_data)
        self._knd.fit(self._metafeatures, self._runs)
        self.fitted = True

    @staticmethod
    def _process_runs(data):
        """
        This function is used to transform the dataframe into a workable object fot the KNN, with rows of pipeline_ids
        and columns of datasets, with the inside being filled with the scores
        :return:
        """
        new_runs = {}
        for index, row in enumerate(data):
            dataset_name = row['dataset_id']
            if dataset_name not in new_runs:
                new_runs[dataset_name] = {}
            new_runs[dataset_name][row['pipeline_id']] = row['test_f1_macro']
        final_new = pd.DataFrame(new_runs)
        return final_new

    @staticmethod
    def _process_metafeatures(data):
        metafeatures = pd.DataFrame(data)
        # Keep just the dataset_id and metafeatures, and expand each
        # metafeature out into its own column.
        metafeatures = pd.concat(
            [metafeatures.dataset_id, metafeatures.metafeatures.apply(pd.Series)],
            axis="columns"
        )
        metafeatures.drop_duplicates(inplace=True)
        metafeatures.set_index("dataset_id", drop=True, inplace=True)
        return metafeatures

class MLPRegressionModel(PyTorchRegressionRankModelBase):

    def __init__(
            self, n_hidden_layers: int, hidden_layer_size: int, activation_name: str, use_batch_norm: bool,
            loss_function_name: str, use_skip: bool = False, dropout = 0.0, *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(device=device, seed=seed, loss_function_name=loss_function_name)

        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation_name = activation_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.output_layer_size = 1

    def _get_model(self, train_data):
        self.input_layer_size = len(train_data[0]['metafeatures'])
        layer_sizes = [self.input_layer_size] + ([self.hidden_layer_size] * self.n_hidden_layers) + \
                      [self.output_layer_size]
        return MLP(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
            seed=self.seed
        )
