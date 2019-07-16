import argparse

import numpy as np
import pandas as pd
import time

from dna import utils
from dna.data import group_json_objects
from dna.metrics import rmse, top_k_regret, top_k_correct, spearman_correlation, pearson_correlation
from dna.models.models import GroupRegressionModel


class ProblemBase:

    def __init__(self):
        self._fit_method_name = 'fit'
        self._predict_method_name = None

    def _validate_model_has_method(self, model, method_name):
        if not hasattr(model, method_name):
            raise ValueError(
                '{} is not designed for the {} problem. It is missing a {} method'.format(
                    model, type(self).__name__, method_name
                )
            )

    def fit(
        self, train_data, test_data, model, model_config, *, refit_model=False, verbose=False, model_output_dir=None
    ):
        self._validate_model_has_method(model, self._fit_method_name)

        model_fit_config = model_config.get(self._fit_method_name, {})
        model_fit_method = getattr(model, self._fit_method_name)

        fit_time = None
        if not model.fitted or refit_model:
            start_time = time.time()
            model_fit_method(
                train_data, validation_data=test_data, verbose=verbose, output_dir=model_output_dir, **model_fit_config
            )
            fit_time = time.time() - start_time

        return fit_time

    def fit_predict(
        self, train_data, test_data, model, model_config, *, refit_model=False, verbose=False, model_output_dir=None
    ):
        fit_time = self.fit(
            train_data, test_data, model, model_config, refit_model=refit_model, verbose=verbose, model_output_dir=model_output_dir
        )

        train_predictions, predict_time = self.predict(
            train_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
        )

        return train_predictions, fit_time, predict_time

    def score(self, predictions, targets):
        raise NotImplementedError()

    def plot(self, predictions, targets, scores):
        raise NotImplementedError()


class RegressionProblem(PredictByGroupProblemBase):

    def __init__(self, group_key):
        super().__init__(group_key)
        self._predict_method_name = 'predict_regression'

    def predict(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        if isinstance(model, GroupRegressionModel):
            return self.predict_by_group(data, model, model_config, verbose=verbose, model_output_dir=model_output_dir)
        else:
            return self.predict_all(data, model, model_config, verbose=verbose, model_output_dir=model_output_dir)

    def predict_by_group(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        return super().predict(data, model, model_config, verbose=verbose, model_output_dir=model_output_dir)

    def predict_all(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        self._validate_model_has_method(model, self._predict_method_name)

        model_predict_config = model_config.get(self._predict_method_name, {})
        model_predict_method = getattr(model, self._predict_method_name)

        start_timestamp = time.time()
        predictions = model_predict_method(data, verbose=verbose, **model_predict_config)
        predict_time = time.time() - start_timestamp

        return predictions, predict_time

    def score(self, predictions, data):
        if type(predictions) == dict:
            return self.score_by_group(predictions, data)
        else:
            return self.score_all(predictions, data)

    def score_by_group(self, predictions_by_group, targets):
        targets_by_group = self._group_data(targets)
        RMSEs = []
        pearson_coefs = []
        pearson_ps = []

        for group, group_predictions in predictions_by_group.items():
            group_predictions = pd.DataFrame(group_predictions)
            group_targets = pd.DataFrame(targets_by_group[group])

            # Make sure no pipelines are being excluded in the merge
            num_pipelines = len(group_targets.pipeline_id)
            assert num_pipelines == len(group_predictions.pipeline_id)
            merged_data = group_targets.merge(group_predictions, on='pipeline_id')
            assert len(merged_data) == num_pipelines

            # TODO: remove hard-coded values
            predictions = merged_data['predictions'].tolist()
            actuals = merged_data['test_f1_macro'].tolist()
            RMSEs.append(rmse(predictions, actuals))
            correlation, p_value = pearson_correlation(predictions, actuals)
            pearson_coefs.append(correlation)
            pearson_ps.append(p_value)

        return {
            'pearson_correlation': {
                'RMSE': np.mean(RMSEs),
                'mean': np.mean(pearson_coefs),
                'std_dev': np.std(pearson_coefs, ddof=1),
                'mean_p_value': np.mean(pearson_ps),
                'std_dev_p_value': np.std(pearson_ps, ddof=1),
            }
        }

    @staticmethod
    def score_all(predictions, data):
        # TODO: just pass in targets
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])

        correlation, p_value = pearson_correlation(predictions, targets)
        return {
            'RMSE': rmse(predictions, targets),
            'PearsonCorrelation': {
                'correlation_coefficient': correlation,
                'p_value': p_value
            }
        }


class PredictByGroupProblemBase(ProblemBase):

    def __init__(self, group_key):
        super().__init__()
        self.group_key = group_key

    def _group_data(self, data):
        grouped_data = {}
        for group, group_indices in group_json_objects(data, self.group_key).items():
            for i in group_indices:
                if group not in grouped_data:
                    grouped_data[group] = []
                grouped_data[group].append(data[i])
        return grouped_data

    def predict(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        self._validate_model_has_method(model, self._predict_method_name)

        model_predict_config = model_config.get(self._predict_method_name, {})
        model_predict_method = getattr(model, self._predict_method_name)

        grouped_data = self._group_data(data)

        start_timestamp = time.time()

        predictions_by_group = {
            group: model_predict_method(group_data, verbose=verbose, **model_predict_config) for group, group_data in grouped_data.items()
        }

        predict_time = time.time() - start_timestamp

        return predictions_by_group, predict_time


class RankProblem(PredictByGroupProblemBase):

    def __init__(self, group_key):
        super().__init__(group_key)
        self._predict_method_name = 'predict_rank'

    def score(self, predictions_by_group, targets):
        targets_by_group = self._group_data(targets)
        spearman_coefs = []
        spearman_ps = []

        for group, group_predictions in predictions_by_group.items():
            group_predictions = pd.DataFrame(group_predictions)
            group_targets = pd.DataFrame(targets_by_group[group])

            # TODO: remove hard-coded values
            merged_data = group_targets.merge(group_predictions, on='pipeline_id')
            correlation, p_value = spearman_correlation(merged_data['rank'], utils.rank(merged_data['test_f1_macro']))
            spearman_coefs.append(correlation)
            spearman_ps.append(p_value)

        return {
            'spearman_correlation': {
                'mean': np.mean(spearman_coefs),
                'std_dev': np.std(spearman_coefs, ddof=1),
                'mean_p_value': np.mean(spearman_ps),
                'std_dev_p_value': np.std(spearman_ps, ddof=1),
            }
        }


class SubsetProblem(PredictByGroupProblemBase):

    def __init__(self, group_key, k):
        super().__init__(group_key)
        self._predict_method_name = 'predict_subset'
        self.k = k

    def predict(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        # TODO: the only difference between this method and PredictByGroupProblemBase's is the use of k
        # How can we remove the duplicate code?
        self._validate_model_has_method(model, self._predict_method_name)

        model_predict_config = model_config.get(self._predict_method_name, {})
        model_predict_method = getattr(model, self._predict_method_name)

        grouped_data = self._group_data(data)

        start_timestamp = time.time()

        predictions_by_group = {
            group: model_predict_method(group_data, k=self.k, verbose=verbose, **model_predict_config) for group, group_data in grouped_data.items()
        }

        predict_time = time.time() - start_timestamp

        return predictions_by_group, predict_time

    def score(self, predictions_by_group, targets):
        targets_by_group = self._group_data(targets)
        top_1_regrets = []
        top_k_regrets = []
        top_k_counts = []

        for group, group_predictions in predictions_by_group.items():
            group_targets = pd.DataFrame(targets_by_group[group])

            top_1_regrets.append(top_k_regret(group_predictions, group_targets, 1))
            top_k_regrets.append(top_k_regret(group_predictions, group_targets, self.k))
            top_k_counts.append(top_k_correct(group_predictions, group_targets, self.k))

        return {
            'top_1_regret': {
                'mean': np.mean(top_1_regrets),
                'std_dev': np.std(top_1_regrets, ddof=1),
            },
            'top_k_regret': {
                'k': self.k,
                'mean': np.mean(top_k_regrets),
                'std_dev': np.std(top_k_regrets, ddof=1),
            },
            'top_k_count': {
                'k': self.k,
                'mean': np.mean(top_k_counts),
                'std_dev': np.std(top_k_counts, ddof=1),
            },
        }


def get_problem(problem_name: str, arguments: argparse.Namespace):
    group_key = 'dataset_id'
    if problem_name == 'regression':
        return RegressionProblem(group_key)
    if problem_name == 'rank':
        return RankProblem(group_key)
    if problem_name == 'subset':
        return SubsetProblem(group_key, arguments.k)
