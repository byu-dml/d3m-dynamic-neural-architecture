import scipy.stats

from metrics import rmse
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
        train_predictions = model_predict_method(train_data, verbose=verbose, **predict_regression_model_config)
        test_predictions = model_predict_method(test_data, verbose=verbose, **predict_regression_model_config)

        train_scores = self._score(train_predictions, train_data)
        test_scores = self._score(test_predictions, test_data)

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
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])
        return {'RMSE': rmse(predictions, targets)}


class RankProblem(ProblemBase):

    _predict_method_name = 'predict_rank'

    @staticmethod
    def _score(predictions, data):
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])
        true_ranks = utils.rank(targets)
        score = scipy.stats.spearmanr(predictions, true_ranks)
        return {'spearman': {'correlation': score.correlation, 'pvalue': score.pvalue}}


class TopKProblem(ProblemBase):

    _predict_method_name = 'predict_rank'

    def run(
        self, train_data, test_data, model, k, *, model_config=None, re_fit_model=False, verbose=False, output_dir=None
    ):
        if self._predict_method_name is None:
            raise NotImplementedError('problem has not been implemented or is not implemented properly')

        if not hasattr(model, 'fit'):
            raise ValueError('The given model is not suited for this problem, it is missing a `fit` method')
        if not hasattr(model, self._predict_method_name):
            raise ValueError('The given model is not suited for this problem, it is missing a `predict_regression` method')

        fit_model_config = model_config.get('fit', {})
        predict_regression_model_config = model_config.get(self._predict_method_name, {})

        if not model.fitted or re_fit_model:
            model.fit(
                train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
            )

        model_predict_method = getattr(model, self._predict_method_name)
        train_predictions = model_predict_method(train_data, k=k, verbose=verbose, **predict_regression_model_config)
        test_predictions = model_predict_method(test_data, k=k, verbose=verbose, **predict_regression_model_config)

        train_scores = self._score(train_predictions, train_data, k)
        test_scores = self._score(test_predictions, test_data, k)

        return train_predictions, test_predictions, train_scores, test_scores

    @staticmethod
    def _score(predictions, data, k):
        pass  # todo


def get_problem(problem_name):
    problem_class = {
        'regression': RegressionProblem,
        'rank': RankProblem,
        'top-k': TopKProblem,
    }[problem_name]
    return problem_class()
