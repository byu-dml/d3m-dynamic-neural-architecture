import numpy as np
import pandas as pd
import time

from data import group_json_objects
from metrics import rmse, top_k_regret, top_k_correct, spearman_correlation, pearson_correlation

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

        start_eval_time_train = time.time()
        train_predictions = model_predict_method(train_data, verbose=verbose, **predict_regression_model_config)
        test_eval_time_train = time.time() - start_eval_time_train

        start_eval_time = time.time()
        test_predictions = model_predict_method(test_data, verbose=verbose, **predict_regression_model_config)
        test_eval_time = time.time() - start_eval_time

        train_scores = self._score(train_predictions, train_data)
        test_scores = self._score(test_predictions, test_data)

        timings = {}
        if fit_time is not None:
            timings["train"] = fit_time
        timings["test_regression_time"] = test_eval_time
        timings["train_regression_time"] = test_eval_time_train

        return train_predictions, test_predictions, train_scores, test_scores, timings

    @staticmethod
    def _score(predictions, data):
        raise NotImplementedError()

    @staticmethod
    def _structure_data(data):
        return data


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
            start_time = time.time()
            model.fit(
                train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
            )
            fit_time = time.time() - start_time

        train_data_by_dataset = self._group_data(train_data)
        test_data_by_dataset = self._group_data(test_data)

        start_eval_time_train = time.time()
        train_predicted_ranks = self._predict_rank(train_data_by_dataset, model, verbose, predict_rank_model_config)
        test_eval_time_train = time.time() - start_eval_time_train

        start_eval_time = time.time()
        test_predicted_ranks = self._predict_rank(test_data_by_dataset, model, verbose, predict_rank_model_config)
        test_eval_time = time.time() - start_eval_time

        train_scores = self._score(scores, train_predicted_ranks, train_data_by_dataset, k)
        test_scores = self._score(scores, test_predicted_ranks, test_data_by_dataset, k)

        timings = {}
        if fit_time is not None:
            timings["train"] = fit_time
        timings["test_rank_time"] = test_eval_time
        timings["train_rank_time"] = test_eval_time_train

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
        spearmans = []
        pearson_coefs = []
        pearson_ps = []
        top_1_regrets = []
        top_k_regrets = []

        for dataset_id, predicted_ranks in predicted_ranks_by_dataset.items():
            predicted_ranks = pd.DataFrame(predicted_ranks)
            actual_ranks = pd.DataFrame(actual_ranks_by_dataset[dataset_id])
            if 'top-k-count' in scores:
                top_k_counts.append(top_k_correct(predicted_ranks, actual_ranks, k))
            if 'spearman' in scores:
                spearmans.append(spearman_correlation(predicted_ranks, actual_ranks))
            if 'pearson' in scores:
                coefficient, p_value = pearson_correlation(pd.DataFrame(predicted_ranks)['rank'], pd.DataFrame(actual_ranks)['test_f1_macro'])
                pearson_coefs.append(coefficient)
                pearson_ps.append(p_value)
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
                'mean': np.mean(spearmans),
                'std_dev': np.std(spearmans, ddof=1),
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
        if 'pearson' in scores:
            results['pearson_correlation'] = {
                'mean': np.mean(pearson_coefs),
                'std_dev': np.std(pearson_coefs, ddof=1),
                'mean_p_value': np.mean(pearson_ps),
                'std_dev_p_value': np.std(pearson_ps, ddof=1),
            }

        return results


def get_problem(problem_name):
    problem_class = {
        'regression': RegressionProblem,
        'rank': RankProblem,
    }[problem_name]
    return problem_class()
