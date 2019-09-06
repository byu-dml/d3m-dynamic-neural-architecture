import argparse
import typing

import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import copy

from dna import utils
from dna.data import group_json_objects
from dna import metrics
from dna import utils


class ProblemBase:

    def __init__(self, group_key, problem_name):
        self.problem_name = problem_name
        self.group_key = group_key
        self.group_scores_key = 'scores_by_{}'.format(self.group_key)
        self.agg_scores_key = 'aggregate_scores'
        self.total_scores_key = 'total_scores'
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

    def __init__(self, group_key):
        super().__init__(group_key, 'regression')
        self._predict_method_name = 'predict_regression'

    def score(self, predictions, data):
        # TODO: just pass in targets
        # Score all the datasets combined
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])
        correlation, p_value = metrics.pearson_correlation(predictions, targets)

        total_scores = {
            'rmse': metrics.rmse(predictions, targets),
            'pearson_correlation': correlation,
            'pearson_p_value': p_value,
        }

        # Score per dataset
        predictions_by_group, targets_by_group = self._group_data(predictions, data)
        scores_by_group, aggregate_scores = self._score_by_group(predictions_by_group, targets_by_group)

        return {
            self.group_scores_key: scores_by_group,
            self.agg_scores_key: aggregate_scores,
            self.total_scores_key: total_scores
        }

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
        scores_by_group = {}

        for group, group_predictions in predictions_by_group.items():
            group_targets = targets_by_group[group]
            RMSE = metrics.rmse(group_predictions, group_targets)
            correlation, p_value = metrics.pearson_correlation(group_predictions, group_targets)

            scores_by_group[group] = {
                'rmse': RMSE,
                'pearson_correlation': correlation,
                'pearson_p_value': p_value,
            }

            RMSEs.append(RMSE)
            pearson_coefs.append(correlation)
            pearson_ps.append(p_value)

        aggregate_scores =  {
            'rmse_mean': np.mean(RMSEs),
            'rmse_std_dev': metrics.std_dev(RMSEs, ddof=1),
            'pearson_correlation_mean': np.mean(pearson_coefs),
            'pearson_correlation_mean_std_dev': metrics.std_dev(pearson_coefs, ddof=1),
            'pearson_p_value_mean': np.mean(pearson_ps),
            'pearson_p_value_std_dev': metrics.std_dev(pearson_ps, ddof=1),
        }

        return scores_by_group, aggregate_scores

    def plot(self, predictions, data, scores, plot_dir: str):
        total_scores, scores_by_group = scores[self.total_scores_key], scores[self.group_scores_key]

        # Plot all the datasets combined
        actuals = [item['test_f1_macro'] for item in data]
        self._plot_base(predictions, actuals, 'All_Datasets', plot_dir, total_scores, type(self).__name__)

        # Plot per dataset
        predictions_by_group, targets_by_group = self._group_data(predictions, data)
        for (group, group_predictions) in predictions_by_group.items():
            group_targets = targets_by_group[group]
            group_scores = scores_by_group[group]
            plot_name = group
            super()._plot_base(group_predictions, group_targets, plot_name, plot_dir, group_scores, type(self).__name__)


class PredictByGroupProblemBase(ProblemBase):

    def __init__(self, group_key, problem_name):
        super().__init__(group_key, problem_name)

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
        super().__init__(group_key, 'rank')
        self._predict_method_name = 'predict_rank'

    @staticmethod
    def _align_predictions_with_targets(group_targets, group_predictions):
        group_targets = pd.DataFrame(group_targets)[['pipeline_id', 'test_f1_macro']]
        group_predictions = pd.DataFrame(group_predictions)[['pipeline_id', 'rank']]
        # left join ensures targets are not dropped if no predictions were made for particular targets
        return group_targets.merge(group_predictions, how='left', on='pipeline_id')

    def _get_scores_by_group(self, targets_by_group, predictions_by_group):
        scores_by_group = {}

        for group, group_predictions in predictions_by_group.items():
            aligned_data = self._align_predictions_with_targets(targets_by_group[group], group_predictions)

            scores_by_group[group] = {}

            correlation, p_value = metrics.spearman_correlation(aligned_data['rank'], utils.rank(aligned_data['test_f1_macro']))
            scores_by_group[group]['spearman_correlation'] = correlation
            scores_by_group[group]['spearman_p_value'] = p_value

            # sorting is an optimization for the @k metrics
            aligned_data.sort_values(['rank', 'pipeline_id'], inplace=True)

            scores_by_group[group]['ndcg_at_k'] = metrics.ndcg_at_k(aligned_data['test_f1_macro'])
            scores_by_group[group]['regret_at_k'] = metrics.regret_at_k(aligned_data['test_f1_macro'])
            scores_by_group[group]['n_correct_at_k'] = metrics.n_correct_at_k(aligned_data['test_f1_macro'])

        return scores_by_group

    def _get_aggregate_scores(self, scores_by_group):
        aggregate_scores = {}
        total_scores = {}

        for score_name in ['spearman_correlation', 'spearman_p_value']:
            scores = list(group_score[score_name] for group, group_score in scores_by_group.items())
            aggregate_scores['{}_mean'.format(score_name)] = np.mean(scores)
            aggregate_scores['{}_std_dev'.format(score_name)] = metrics.std_dev(scores, ddof=1)

        for score_name in ['ndcg_at_k', 'regret_at_k', 'n_correct_at_k']:
            scores = (group_score[score_name] for group, group_score in scores_by_group.items())
            scores_by_k = utils.transpose_jagged_2darray(scores)
            aggregate_scores['{}_mean'.format(score_name)] = [np.mean(scores_at_k) for i, scores_at_k in scores_by_k.items()]
            aggregate_scores['{}_std_dev'.format(score_name)] = [metrics.std_dev(scores_at_k, ddof=1) for i, scores_at_k in scores_by_k.items()]

            flattened_scores = [score_at_k for i, scores_at_k in scores_by_k.items() for score_at_k in scores_at_k]
            total_scores['{}_mean'.format(score_name)] = np.mean(flattened_scores)
            total_scores['{}_std_dev'.format(score_name)] = metrics.std_dev(flattened_scores, ddof=1)

        return aggregate_scores, total_scores


    def score(self, predictions_by_group, targets):
        """Computes Spearman correlation, ndcg_at_k, regret_at_k, n_correct_at_k"""
        targets_by_group = self._group_data(targets)
        scores_by_group = self._get_scores_by_group(targets_by_group, predictions_by_group)
        aggregate_scores, total_scores = self._get_aggregate_scores(scores_by_group)
        return {self.group_scores_key: scores_by_group, self.agg_scores_key: aggregate_scores, self.total_scores_key: total_scores}

    def plot(self, predictions, targets, scores, plot_dir: str):
        scores_by_group = scores[self.group_scores_key]
        grouped_targets = self._group_data(targets)

        for (dataset_id, predicted_ranks) in predictions.items():
            predicted_ranks = pd.DataFrame(predicted_ranks)
            actuals_by_dataset = pd.DataFrame(grouped_targets[dataset_id])
            merged_data = predicted_ranks.merge(actuals_by_dataset, on='pipeline_id')
            predicted_ranks = merged_data['rank'].tolist()
            predicted_ranks = np.array(predicted_ranks)
            actuals = merged_data['test_f1_macro'].tolist()
            actual_ranks = utils.rank(actuals)

            # Average all the scores at k to make the plot title a reasonable length
            group_scores = scores_by_group[dataset_id]
            group_scores = self.shorten_k_rank_scores(group_scores)

            plot_name = dataset_id + '_plot'
            super()._plot_base(predicted_ranks, actual_ranks, plot_name, plot_dir, group_scores, type(self).__name__)

    @staticmethod
    def shorten_k_rank_scores(rank_scores: dict):
        """Takes the average of the top k, k regret, and ndcg rank scores so the score at every k isn't reported"""

        rank_scores = copy.deepcopy(rank_scores)
        rank_scores['ndcg_at_k'] = np.mean(rank_scores['ndcg_at_k'])
        rank_scores['regret_at_k'] = np.mean(rank_scores['regret_at_k'])
        rank_scores['n_correct_at_k'] = np.mean(rank_scores['n_correct_at_k'])
        return rank_scores


def get_problem(problem_name: str, **kwargs):
    group_key = 'dataset_id'
    if problem_name == 'regression':
        return RegressionProblem(group_key)
    if problem_name == 'rank':
        return RankProblem(group_key)
