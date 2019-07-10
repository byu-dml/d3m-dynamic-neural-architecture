import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from dna import utils
from dna.data import group_json_objects
from dna.metrics import rmse, top_k_regret, top_k_correct, spearman_correlation, pearson_correlation
from dna import utils

class ProblemBase:

    _predict_method_name = None

    def run(
        self, train_data, test_data, model, *, model_config=None, re_fit_model=False, verbose=False, output_dir=None
    ):
        if self._predict_method_name is None:
            raise NotImplementedError('problem has not been implemented or is not implemented properly')

        if not hasattr(model, 'fit'):
            raise ValueError('The given model is not suited for this problem, it is missing a `fit` method')
        if not hasattr(model, self._predict_method_name):
            raise ValueError('The given model is not suited for this problem, it is missing a `predict_regression` method')

        train_data = self._structure_data(train_data)
        test_data = self._structure_data(test_data)

        fit_model_config = model_config.get('fit', {})
        predict_regression_model_config = model_config.get(self._predict_method_name, {})

        fit_time = None
        if not model.fitted or re_fit_model:
            start_time = time.time()
            model.fit(
                train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
            )
            fit_time = time.time() - start_time


        model_predict_method = getattr(model, self._predict_method_name)

        start_timestamp = time.time()
        train_predictions = model_predict_method(train_data, verbose=verbose, **predict_regression_model_config)
        train_predict_time = time.time() - start_timestamp

        start_timestamp = time.time()
        test_predictions = model_predict_method(test_data, verbose=verbose, **predict_regression_model_config)
        test_predict_time = time.time() - start_timestamp

        train_scores = self._score(train_predictions, train_data)
        self._plot(train_predictions, train_data, plot_name='train', plot_directory=output_dir, scores=train_scores,
                   problem_name=str(type(self)))
        test_scores = self._score(test_predictions, test_data)
        self._plot(test_predictions, test_data, plot_name='validation', plot_directory=output_dir, scores=test_scores,
                   problem_name=str(type(self)))

        timings = {
            'fit_time': fit_time,
            'train_predict_time': train_predict_time,
            'test_predict_time': test_predict_time,
        }

        return train_predictions, test_predictions, train_scores, test_scores, timings

    @staticmethod
    def _score(predictions, data):
        raise NotImplementedError()

    @staticmethod
    def _structure_data(data):
        return data

    def _plot(self, predictions, data, plot_name: str, plot_directory: str, scores: dict, problem_name: str):
        actuals = [item['test_f1_macro'] for item in data]
        actuals = np.array(actuals)
        self._plot_base(predictions, actuals, plot_name, plot_directory, scores, problem_name)

    @staticmethod
    def _plot_base(predictions, actuals, plot_name: str, plot_directory: str, scores: dict, problem_name: str):
        if type(predictions) == list:
            predictions = np.array(predictions)
        if type(actuals) == list:
            actuals = np.array(actuals)

        if(len(predictions) != len(actuals)):
            print('The length of the predictions must match the length of the actuals')
            return

        # Create the title with the scores on it
        title = ProblemBase._make_plot_title('', scores)
        plt.title(title, fontsize=6)

        # Set the min and max value on the x and y axis
        predictions_min = predictions.min()
        actuals_min = actuals.min()
        axis_min = min(actuals_min, predictions_min)
        predictions_max = predictions.max()
        actuals_max = actuals.max()
        axis_max = max(actuals_max, predictions_max)
        plt.xlim(axis_min, axis_max)
        plt.ylim(axis_min, axis_max)

        # Create the plot
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.scatter(predictions, actuals)
        plt.tight_layout()

        # Save the plot
        new_dir = os.path.join(plot_directory, problem_name)
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        file_name = os.path.join(new_dir, plot_name)
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

    _predict_method_name = 'predict_regression'

    @staticmethod
    def _score(predictions, data):
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])

        correlation, p_value = pearson_correlation(predictions, targets)

        return {'RMSE': rmse(predictions, targets), 'PearsonCorrelation': {'correlation_coefficient': correlation, 'p_value': p_value}}


class RankProblem(ProblemBase):

    def run(
        self, train_data, test_data, model, k, scores, *, model_config=None, re_fit_model=False, verbose=False, output_dir=None
    ):
        # todo multiple k_s
        if not hasattr(model, 'fit'):
            raise ValueError('The given model is not suited for this problem, it is missing a `fit` method')
        if not hasattr(model, 'predict_rank'):
            raise ValueError('The given model is not suited for this problem, it is missing a `predict_regression` method')

        train_data = self._structure_data(train_data)
        test_data = self._structure_data(test_data)

        fit_model_config = model_config.get('fit', {})
        predict_rank_model_config = model_config.get('predict_rank', {})

        fit_time = None
        if not model.fitted or re_fit_model:
            start_timestamp = time.time()
            model.fit(
                train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
            )
            fit_time = time.time() - start_timestamp

        train_data_by_dataset = self._group_data(train_data)
        test_data_by_dataset = self._group_data(test_data)

        start_timestamp = time.time()
        train_predicted_ranks = self._predict_rank(train_data_by_dataset, model, verbose, predict_rank_model_config)
        train_predict_time = time.time() - start_timestamp

        start_timestamp = time.time()
        test_predicted_ranks = self._predict_rank(test_data_by_dataset, model, verbose, predict_rank_model_config)
        test_predict_time = time.time() - start_timestamp

        train_scores = self._score(scores, train_predicted_ranks, train_data_by_dataset, k)
        self._plot(train_predicted_ranks, train_data_by_dataset, plot_name='train', plot_directory=output_dir,
                   scores=train_scores, problem_name=str(type(self)))
        test_scores = self._score(scores, test_predicted_ranks, test_data_by_dataset, k)
        self._plot(test_predicted_ranks, test_data_by_dataset, plot_name='validation', plot_directory=output_dir,
                   scores=test_scores, problem_name=str(type(self)))

        timings = {
            'fit_time': fit_time,
            'train_predict_time': train_predict_time,
            'test_predict_time': test_predict_time,
        }

        return train_predicted_ranks, test_predicted_ranks, train_scores, test_scores, timings

    @staticmethod
    def _group_data(data):
        grouped_data = {}
        for dataset_id, group_indices in group_json_objects(data, 'dataset_id').items():
            for i in group_indices:
                if dataset_id not in grouped_data:
                    grouped_data[dataset_id] = []
                grouped_data[dataset_id].append(data[i])
        return grouped_data

    @staticmethod
    def _predict_rank(data_by_dataset, model, verbose, predict_rank_model_config):
        predictions_by_dataset = {}
        for dataset_id, data_subset in data_by_dataset.items():
            predicted_ranks = model.predict_rank(data_subset, verbose=verbose, **predict_rank_model_config)
            predictions_by_dataset[dataset_id] = predicted_ranks
        return predictions_by_dataset

    @staticmethod
    def _score(scores, predicted_ranks_by_dataset: dict, actual_ranks_by_dataset: dict, k):
        if not scores:
            return {}

        top_k_counts = []
        spearman_coefs = []
        spearman_ps = []
        top_1_regrets = []
        top_k_regrets = []

        for dataset_id, predicted_ranks in predicted_ranks_by_dataset.items():
            predicted_ranks = pd.DataFrame(predicted_ranks)
            actual_ranks = pd.DataFrame(actual_ranks_by_dataset[dataset_id])
            merged_data = actual_ranks.merge(predicted_ranks, on='pipeline_id')
            if 'top-k-count' in scores:
                top_k_counts.append(top_k_correct(predicted_ranks, actual_ranks, k))
            if 'spearman' in scores:
                correlation, p_value = spearman_correlation(merged_data['rank'], utils.rank(merged_data['test_f1_macro']))
                spearman_coefs.append(correlation)
                spearman_ps.append(p_value)
            if 'top-1-regret' in scores:
                top_1_regrets.append(top_k_regret(predicted_ranks, actual_ranks, 1))
            if 'top-k-regret' in scores:
                top_k_regrets.append(top_k_regret(predicted_ranks, actual_ranks, k))

        results = {}
        if 'top-k-count' in scores:
            results['top_k_count'] = {
                'k': k,
                'mean': np.mean(top_k_counts),
                'std_dev': np.std(top_k_counts, ddof=1),
            }
        if 'spearman' in scores:
            results['spearman_correlation'] = {
                'mean': np.mean(spearman_coefs),
                'std_dev': np.std(spearman_coefs, ddof=1),
                'mean_p_value': np.mean(spearman_ps),
                'std_dev_p_value': np.std(spearman_ps, ddof=1),
            }
        if 'top-1-regret' in scores:
            results['top_1_regret'] = {
                'mean': np.mean(top_1_regrets),
                'std_dev': np.std(top_1_regrets, ddof=1),
            }
        if 'top-k-regret' in scores:
            results['top_k_regret'] = {
                'k': k,
                'mean': np.mean(top_k_regrets),
                'std_dev': np.std(top_k_regrets, ddof=1),
            }

        return results

    def _plot(self, predictions, data, plot_name: str, plot_directory: str, scores: dict, problem_name: str):
        for (dataset_id, predicted_ranks) in predictions.items():
            predicted_ranks = pd.DataFrame(predicted_ranks)
            predicted_ranks = predicted_ranks['rank'].tolist()
            predicted_ranks = np.array(predicted_ranks)
            actuals_by_dataset = pd.DataFrame(data[dataset_id])
            actuals = actuals_by_dataset['test_f1_macro'].tolist()
            actual_ranks = utils.rank(actuals)
            plot_name = plot_name + '_' + dataset_id

            ProblemBase._plot_base(predicted_ranks, actual_ranks, plot_name, plot_directory, scores, problem_name)


def get_problem(problem_name):
    problem_class = {
        'regression': RegressionProblem,
        'rank': RankProblem,
    }[problem_name]
    return problem_class()
