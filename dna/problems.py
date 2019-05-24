import numpy as np
import scipy.stats

from data import group_json_objects
from metrics import rmse, regret_value, top_k, spearman_correlation
import utils


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

        if not model.fitted or re_fit_model:
            model.fit(
                train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
            )

        model_predict_method = getattr(model, self._predict_method_name)
        train_predictions, train_targets = model_predict_method(train_data, verbose=verbose, **predict_regression_model_config)
        test_predictions, test_targets = model_predict_method(test_data, verbose=verbose, **predict_regression_model_config)

        train_scores = self._score(train_predictions, train_targets)
        test_scores = self._score(test_predictions, test_targets)

        return train_predictions, test_predictions, train_scores, test_scores

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
        # targets = []
        # for instance in data:
        #     targets.append(instance['test_f1_macro'])
        return {'RMSE': rmse(predictions, data)}



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

        if not model.fitted or re_fit_model:
            model.fit(
                train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
            )

        train_data_by_dataset = self._group_data(train_data)
        test_data_by_dataset = self._group_data(test_data)

        train_predicted_ranks = self._predict_rank(train_data_by_dataset, model, verbose, predict_rank_model_config)
        test_predicted_ranks = self._predict_rank(test_data_by_dataset, model, verbose, predict_rank_model_config)

        train_scores = self._score(scores, train_predicted_ranks, train_data_by_dataset, k)
        test_scores = self._score(scores, test_predicted_ranks, test_data_by_dataset, k)

        return train_predicted_ranks, test_predicted_ranks, train_scores, test_scores

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
        regrets = []

        for dataset_id, predicted_ranks in predicted_ranks_by_dataset.items():
            actual_ranks = actual_ranks_by_dataset[dataset_id]
            if 'top-k-count' in scores:
                top_k_counts.append(top_k(predicted_ranks, actual_ranks, k))
            if 'spearman' in scores:
                spearmans.append(spearman_correlation(predicted_ranks, actual_ranks))
            if 'top-1-regret' in scores:
                regrets.append(regret_value(predicted_ranks, actual_ranks))

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
                'mean': np.mean(regrets),
                'std_dev': np.std(regrets, ddof=1),
            }
        return results


def get_problem(problem_name):
    problem_class = {
        'regression': RegressionProblem,
        'rank': RankProblem,
    }[problem_name]
    return problem_class()
