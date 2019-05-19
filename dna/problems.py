from metrics import rmse


class ProblemBase:

    def run(self, train_data, test_data, model):
        raise NotImplementedError()


class RegressionProblem(ProblemBase):

    def run(
        self, train_data, test_data, model, *, model_config=None,
        re_fit_model=False, verbose=False, output_dir=None
    ):
        if not hasattr(model, 'fit'):
            raise ValueError('The given model is not suited for this problem, it is missing a `fit` method')
        if not hasattr(model, 'predict_regression'):
            raise ValueError('The given model is not suited for this problem, it is missing a `predict_regression` method')

        fit_model_config = model_config.get('fit', {})
        predict_regression_model_config = model_config.get('predict_regression', {})

        # todo handle re_fit
        model.fit(
            train_data, validation_data=test_data, verbose=verbose, output_dir=output_dir, **fit_model_config
        )

        train_predictions = model.predict_regression(train_data, verbose=verbose, **predict_regression_model_config)
        test_predictions = model.predict_regression(test_data, verbose=verbose, **predict_regression_model_config)

        train_score = self._score(train_predictions, train_data)
        test_score = self._score(test_predictions, test_data)

        return train_predictions, test_predictions, train_score, test_score

    @staticmethod
    def _score(predictions, data):
        targets = []
        for instance in data:
            targets.append(instance['test_f1_macro'])
        return rmse(predictions, targets)


def get_problem(problem_name):
    problem_class = {
        'regression': RegressionProblem,
    }[problem_name]
    return problem_class()
