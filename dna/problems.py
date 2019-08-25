import argparse

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from dna import utils
from dna.data import group_json_objects
from dna.metrics import rmse, top_k_regret, top_k_correct, spearman_correlation, pearson_correlation, ndcg_at_k
from dna import utils


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
        self, train_data, model, model_config, *, refit_model=False, verbose=False, model_output_dir=None
    ):
        self._validate_model_has_method(model, self._fit_method_name)

        model_fit_config = model_config.get(self._fit_method_name, {})
        model_fit_method = getattr(model, self._fit_method_name)

        fit_time = None
        if not model.fitted or refit_model:
            start_time = time.time()
            model_fit_method(
                train_data, verbose=verbose, output_dir=model_output_dir, **model_fit_config
            )
            fit_time = time.time() - start_time

        return fit_time

    def predict(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        self._validate_model_has_method(model, self._predict_method_name)

        model_predict_config = model_config.get(self._predict_method_name, {})
        model_predict_method = getattr(model, self._predict_method_name)

        start_timestamp = time.time()
        predictions = model_predict_method(data, verbose=verbose, **model_predict_config)
        predict_time = time.time() - start_timestamp

        return predictions, predict_time

    def fit_predict(
        self, train_data, model, model_config, *, refit_model=False, verbose=False, model_output_dir=None
    ):
        fit_time = self.fit(
            train_data, model, model_config, refit_model=refit_model, verbose=verbose, model_output_dir=model_output_dir
        )

        train_predictions, predict_time = self.predict(
            train_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
        )

        return train_predictions, fit_time, predict_time

    def score(self, predictions, targets):
        raise NotImplementedError()

    def plot(self, predictions, targets, scores, plot_dir: str):
        raise NotImplementedError()

    @staticmethod
    def _plot_base(predictions, actuals, plot_name: str, plot_directory: str, scores: dict, problem_name: str):
        if type(predictions) == list:
            predictions = np.array(predictions)
        if type(actuals) == list:
            actuals = np.array(actuals)

        if(len(predictions) != len(actuals)):
            raise ValueError('The length of the predictions must match the length of the actuals')

        # Create the title with the scores on it
        title = ProblemBase._make_plot_title('', scores)
        plt.title(title, fontsize=6)

        # Create the plot
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.scatter(predictions, actuals)
        plt.tight_layout()

        # Save the plot
        new_dir = os.path.join(plot_directory, problem_name)
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        file_name = os.path.join(new_dir, plot_name + '.pdf')
        plt.savefig(fname=file_name)
        plt.clf()

    @staticmethod
    def _make_plot_title(title, scores):
        for (score_name, score_value) in scores.items():
            if type(score_value) == dict:
                title += score_name.upper() + ':' + '\n'
                title = ProblemBase._make_plot_title(title, score_value)
            elif type(score_value) == np.float64 or type(score_value) == float:
                score_value = '{0:.5f}'.format(score_value)
                title += score_name + ': ' + score_value + '\n'
            else:
                title += score_name + ': ' + str(score_value) + '\n'
        return title


class RegressionProblem(ProblemBase):

    def __init__(self):
        super().__init__()
        self.group_key = 'dataset_id'
        self._predict_method_name = 'predict_regression'

    def score(self, predictions, data):
        # TODO: just pass in targets
        # Score all the datasets combined
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])
        correlation, p_value = pearson_correlation(predictions, targets)

        total_scores = {
            'total_rmse': rmse(predictions, targets),
            'total_pearson_correlation': {
                'correlation_coefficient': correlation,
                'p_value': p_value
            }
        }

        # Score per dataset
        predictions_by_group, targets_by_group = self._group_data(predictions, data)
        per_dataset_scores, mean_scores = self._score_by_group(predictions_by_group, targets_by_group)

        return {'total_scores': total_scores, 'per_dataset_scores': per_dataset_scores, 'mean_scores': mean_scores}

    def _group_data(self, predictions, data):
        predictions_by_group = {}
        targets_by_group = {}
        for group, group_indices in group_json_objects(data, self.group_key).items():
            for i in group_indices:
                if group not in predictions_by_group:
                    predictions_by_group[group] = []
                    targets_by_group[group] = []
                predictions_by_group[group].append(predictions[i])
                targets_by_group[group].append(data[i]['test_f1_macro'])
        return predictions_by_group, targets_by_group

    @staticmethod
    def _score_by_group(predictions_by_group, targets_by_group):
        RMSEs = []
        pearson_coefs = []
        pearson_ps = []
        per_dataset_scores = {}

        for group, group_predictions in predictions_by_group.items():
            group_targets = targets_by_group[group]
            RMSE = rmse(group_predictions, group_targets)
            correlation, p_value = pearson_correlation(group_predictions, group_targets)

            per_dataset_scores[group] = {
                'rmse': RMSE,
                'pearson_correlation': {
                    'correlation_coefficient': correlation,
                    'p_value': p_value
                }
            }

            RMSEs.append(RMSE)
            pearson_coefs.append(correlation)
            pearson_ps.append(p_value)

        mean_scores =  {
            'rmse': {
                'mean': np.mean(RMSEs),
                'std_dev': np.std(RMSEs, ddof=1)
            },
            'pearson_correlation': {
                'mean': np.mean(pearson_coefs),
                'std_dev': np.std(pearson_coefs, ddof=1),
                'mean_p_value': np.mean(pearson_ps),
                'std_dev_p_value': np.std(pearson_ps, ddof=1),
            }
        }

        return per_dataset_scores, mean_scores

    def plot(self, predictions, data, scores, plot_dir: str):
        total_scores, per_dataset_scores = scores['total_scores'], scores['per_dataset_scores']

        # Plot all the datasets combined
        actuals = [item['test_f1_macro'] for item in data]
        self._plot_base(predictions, actuals, 'All_Datasets', plot_dir, total_scores, type(self).__name__)

        # Plot per dataset
        predictions_by_group, targets_by_group = self._group_data(predictions, data)
        for (group, group_predictions) in predictions_by_group.items():
            group_targets = targets_by_group[group]
            group_scores = per_dataset_scores[group]
            plot_name = group
            super()._plot_base(group_predictions, group_targets, plot_name, plot_dir, group_scores, type(self).__name__)


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
        ndcg_list = []
        per_dataset_scores = {}

        for group, group_predictions in predictions_by_group.items():
            group_predictions = pd.DataFrame(group_predictions)
            # have to sort by id in cases of ties
            group_targets = pd.DataFrame(targets_by_group[group])[['pipeline_id', 'test_f1_macro']]
            group_targets.sort_values(['test_f1_macro', 'pipeline_id'], ascending=False, inplace=True)
            group_predictions.sort_values(['rank'], ascending=True, inplace=True)

            # get IR metrics
            ndcg_value = ndcg_at_k(group_targets['test_f1_macro'], utils.rank(group_predictions['rank']))

            # TODO: remove hard-coded values
            merged_data = group_targets.merge(group_predictions, on='pipeline_id')
            correlation, p_value = spearman_correlation(merged_data['rank'], utils.rank(merged_data['test_f1_macro']))

            per_dataset_scores[group] = {
                'spearman_correlation': {
                        'correlation_coefficient': correlation,
                        'p_value': p_value
                    },
                'ndcg': ndcg_value,
            }

            spearman_coefs.append(correlation)
            spearman_ps.append(p_value)
            ndcg_list.append(ndcg_value)

        mean_scores = {
            'spearman_correlation': {
                'mean': np.mean(spearman_coefs),
                'std_dev': np.std(spearman_coefs, ddof=1),
                'mean_p_value': np.mean(spearman_ps),
                'std_dev_p_value': np.std(spearman_ps, ddof=1),
            },
            'ndcg': np.mean(ndcg_list),
        }

        return {'per_dataset_scores': per_dataset_scores, 'mean_scores': mean_scores}

    def plot(self, predictions, targets, scores, plot_dir: str):
        per_dataset_scores = scores['per_dataset_scores']
        grouped_targets = self._group_data(targets)

        for (dataset_id, predicted_ranks) in predictions.items():
            predicted_ranks = pd.DataFrame(predicted_ranks)
            actuals_by_dataset = pd.DataFrame(grouped_targets[dataset_id])
            merged_data = predicted_ranks.merge(actuals_by_dataset, on='pipeline_id')
            predicted_ranks = merged_data['rank'].tolist()
            predicted_ranks = np.array(predicted_ranks)
            actuals = merged_data['test_f1_macro'].tolist()
            actual_ranks = utils.rank(actuals)
            group_scores = per_dataset_scores[dataset_id]
            plot_name = dataset_id + '_plot'
            super()._plot_base(predicted_ranks, actual_ranks, plot_name, plot_dir, group_scores, type(self).__name__)


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
        ndcg_at_ks = []

        for group, group_predictions in predictions_by_group.items():
            group_targets = pd.DataFrame(targets_by_group[group])
            # have to sort by id in cases of ties
            group_targets = pd.DataFrame(targets_by_group[group])[['pipeline_id', 'test_f1_macro']]
            group_targets.sort_values(['test_f1_macro', 'pipeline_id'], ascending=False, inplace=True)

            top_1_regrets.append(top_k_regret(group_predictions, group_targets, 1))
            top_k_regrets.append(top_k_regret(group_predictions, group_targets, self.k))
            top_k_counts.append(top_k_correct(group_predictions, group_targets, self.k))
            ndcg_at_ks.append(self._ndcg_score(group_predictions, group_targets))

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
            'ndcg_at_k': {
                'k': self.k,
                'mean': np.mean(ndcg_at_ks),
                'std_dev': np.std(ndcg_at_ks, ddof=1),
            }
        }

    def plot(self, *args, **kwargs):
        pass

    def _ndcg_score(self, group_predictions, group_targets):
        rank_map = {pipeline_id: i for i, pipeline_id in enumerate(group_predictions)}
        relevance = []
        rank = []
        for i, (idx, row) in enumerate(group_targets.iterrows()):
            relevance.append(row['test_f1_macro'])
            rank.append(rank_map.get(row['pipeline_id'], i + self.k))
        return ndcg_at_k(relevance, rank, k=self.k)


def get_problem(problem_name: str, **kwargs):
    group_key = 'dataset_id'
    if problem_name == 'regression':
        return RegressionProblem()
    if problem_name == 'rank':
        return RankProblem(group_key)
    if problem_name == 'subset':
        return SubsetProblem(group_key, kwargs['k'])
