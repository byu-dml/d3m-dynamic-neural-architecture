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

    ## old spearman code:
    # def get_correlation_coefficient(self, dataloader):
    #     # TODO: Handle ties
    #     dataset_performances = {}
    #     pipeline_key = 'pipeline_ids'
    #     actual_key = 'f1_actuals'
    #     predict_key = 'f1_predictions'
    #     for x_batch, y_batch in dataloader:
    #         y_hat_batch = self.model(x_batch)

    #         # Get the pipeline id and the data set ids that correspond to it
    #         pipeline_id, pipeline, x, dataset_ids = x_batch

    #         # Create a list of tuples containing the pipeline id and its f1 values for each data set in this batch
    #         for i in range(len(dataset_ids)):
    #             dataset_id = dataset_ids[i]
    #             f1_actual = y_batch[i].item()
    #             f1_predict = y_hat_batch[i].item()
    #             if dataset_id in dataset_performances:
    #                 dataset_performance = dataset_performances[dataset_id]
    #                 pipeline_ids = dataset_performance[pipeline_key]
    #                 f1_actuals = dataset_performance[actual_key]
    #                 f1_predictions = dataset_performance[predict_key]
    #                 pipeline_ids.append(pipeline_id)
    #                 f1_actuals.append(f1_actual)
    #                 f1_predictions.append(f1_predict)
    #             else:
    #                 dataset_performance = {pipeline_key: [pipeline_id], actual_key: [f1_actual], predict_key: [f1_predict]}
    #                 dataset_performances[dataset_id] = dataset_performance

    #     dataset_cc_sum = 0.0
    #     dataset_performances = dataset_performances.values()
    #     for dataset_performance in dataset_performances:
    #         f1_actuals = dataset_performance[actual_key]
    #         f1_predictions = dataset_performance[predict_key]
    #         actual_ranks = self.rank(f1_actuals)
    #         predicted_ranks = self.rank(f1_predictions)

    #         # Get the spearman correlation coefficient for this data set
    #         spearman_result = scipy.stats.spearmanr(actual_ranks, predicted_ranks)
    #         dataset_cc = spearman_result.correlation
    #         dataset_cc_sum += dataset_cc
    #     num_datasets = len(dataset_performances)
    #     mean_dataset_cc = dataset_cc_sum / num_datasets
    #     return mean_dataset_cc


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
