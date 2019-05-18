import numpy as np
import scipy.stats
import torch
import numpy as np

from data import TRAIN_DATA_PATH, TEST_DATA_PATH
from .base_problem import BaseProblem
from metrics import rmse
from models import Submodule, DNAModel


# TODO: make this dynamic
lookup_input_size = {
    "d3m.primitives.data_transformation.construct_predictions.DataFrameCommon": 2,
    "d3m.primitives.classification.gaussian_naive_bayes.SKlearn": 2,
    'd3m.primitives.classification.linear_svc.SKlearn': 2,
    "d3m.primitives.classification.random_forest.SKlearn": 2,
    "d3m.primitives.classification.gradient_boosting.SKlearn": 2,
    "d3m.primitives.classification.bagging.SKlearn": 2,
    "d3m.primitives.classification.bernoulli_naive_bayes.SKlearn": 2,
    "d3m.primitives.classification.decision_tree.SKlearn": 2,
    "d3m.primitives.classification.k_neighbors.SKlearn": 2,
    "d3m.primitives.classification.linear_discriminant_analysis.SKlearn": 2,
    "d3m.primitives.classification.logistic_regression.SKlearn": 2,
    "d3m.primitives.classification.linear_svc.SKlearn": 2,
    "d3m.primitives.classification.sgd.SKlearn": 2,
    "d3m.primitives.classification.svc.SKlearn": 2,
    "d3m.primitives.classification.extra_trees.SKlearn": 2,
    "d3m.primitives.classification.passive_aggressive.SKlearn": 2,
    "d3m.primitives.feature_selection.select_fwe.SKlearn": 2,
    "d3m.primitives.feature_selection.select_percentile.SKlearn": 2,
    "d3m.primitives.feature_selection.generic_univariate_select.SKlearn": 2,
    'd3m.primitives.regression.extra_trees.SKlearn': 2,
    "d3m.primitives.regression.svr.SKlearn": 2,
    'd3m.primitives.data_transformation.horizontal_concat.DataFrameConcat': 2,

}


class ModelNotFitError(Exception):
    pass


class MeanBaseline:

    def __init__(self):
        self.mean = None

    def fit(self, train_data, target_key):
        total = 0
        for instance in train_data:
            total += instance[target_key]
        self.mean = total / len(train_data)

    def predict(self, data):
        if self.mean is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.mean] * len(data)


class MedianBaseline:

    def __init__(self):
        self.median = None

    def fit(self, train_data, target_key):
        self.median = np.median([instance[target_key] for instance in train_data])

    def predict(self, data):
        if self.median is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.median] * len(data)


class PerPrimitiveBaseline:

    def __init__(self):
        self.primitive_scores = None

    def fit(self, train_data, target_key):
        # for each primitive, get the scores of all the pipelines that use the primitive
        primitive_score_totals = {}
        for instance in train_data:
            for primitive in instance['pipeline']:
                if primitive['name'] not in primitive_score_totals:
                    primitive_score_totals[primitive['name']] = {
                        'total': 0,
                        'count': 0,
                    }
                primitive_score_totals[primitive['name']]['total'] += instance[target_key]
                primitive_score_totals[primitive['name']]['count'] += 1

        # compute the average pipeline score per primitive
        self.primitive_scores = {}
        for primitive_name in primitive_score_totals:
            total = primitive_score_totals[primitive_name]['total']
            count = primitive_score_totals[primitive_name]['count']
            self.primitive_scores[primitive_name] = total / count

    def predict(self, data):
        if self.primitive_scores is None:
            raise ModelNotFitError('PerPrimitiveBaseline not fit')

        predictions = []
        for instance in data:
            prediction = 0
            for primitive in instance['pipeline']:
                prediction += self.primitive_scores[primitive['name']]
            prediction /= len(instance['pipeline'])
            predictions.append(prediction)

        return predictions


class Regression(BaseProblem):

    def __init__(
        self, train_data_path: str = TRAIN_DATA_PATH,
        test_data_path: str = TEST_DATA_PATH, n_folds: int = 5,
        batch_size = 32, drop_last = False, device = "cuda:0", seed = 0
    ):
        self._target_key = "test_accuracy"
        objective = torch.nn.MSELoss(reduction="mean")
        self._loss_function = lambda y, y_hat: torch.sqrt(objective(y, y_hat))
        super(Regression, self).__init__(
            train_data_path = train_data_path,
            test_data_path = test_data_path,
            batch_group_key = "pipeline_id",
            target_key = self._target_key,
            task_type = "REGRESSION",
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed
        )
        self._shape = (len(self._train_data[0]["metafeatures"]), 1)
        self._init_model()

    def _init_model(self):
        torch_state = torch.random.get_rng_state()

        torch.manual_seed(self._randint())
        torch.cuda.manual_seed_all(self._randint())
        input_model = Submodule("input", self._shape[0], self._shape[0])
        input_model.cuda()
        submodels = {}
        for item in self._train_data:
            primitive_names = [dict_obj["name"] for dict_obj in item["pipeline"]]
            for primitive_name in primitive_names:
                if not primitive_name in submodels:
                    try:
                        n_inputs = lookup_input_size[primitive_name]
                    except KeyError as e:
                        n_inputs = 1
                    submodels[primitive_name] = Submodule(
                        primitive_name, n_inputs * self._shape[0], self._shape[0]
                    )
                    submodels[primitive_name].cuda()
        output_model = Submodule('task', self._shape[0], 1, use_skip=False)
        output_model.cuda()
        self._model = DNAModel(input_model, submodels, output_model)
        if "cuda" in self.device:
            self._model.cuda()
        torch.random.set_rng_state(torch_state)

    def _process_train_data(self):
        pass

    def _process_test_data(self):
        pass

    def _compute_baselines(self):
        train_accuracies = []
        for x_batch, y_batch in self._train_data_loader:
            train_accuracies.extend(y_batch.tolist())
        train_accuracies = np.array(train_accuracies)
        train_mean = np.mean(train_accuracies)
        train_median = np.median(train_accuracies)
        mean_rmse = np.sqrt(np.mean((train_accuracies - train_mean)**2))
        median_rmse = np.sqrt(np.mean((train_accuracies - train_median)**2))
        guess_accuracy = train_mean
        train_rmse = mean_rmse
        if median_rmse < mean_rmse:
            guess_accuracy = train_median
            train_rmse = median_rmse

        validation_accuracies = []
        for x_batch, y_batch in self._validation_data_loader:
            validation_accuracies.extend(y_batch.tolist())
        validation_accuracies = np.array(validation_accuracies)
        validation_rmse = np.sqrt(np.mean((validation_accuracies - guess_accuracy)**2))

        self._baselines = {
            'MeanBaselineRMSE': self._compute_baseline(MeanBaseline),
            'MedianBaselineRMSE': self._compute_baseline(MedianBaseline),
            'PerPrimitiveBaselineRMSE': self._compute_baseline(PerPrimitiveBaseline),
        }

    def _compute_baseline(self, baseline_class):
        baseline = baseline_class()
        baseline.fit(self.train_data, self._target_key)
        train_predictions = baseline.predict(self.train_data)
        train_targets = [instance[self._target_key] for instance in self.train_data]
        val_predictions = baseline.predict(self.validation_data)
        val_targets = [instance[self._target_key] for instance in self.validation_data]
        return {
            'train': rmse(train_predictions, train_targets),
            'validation': rmse(val_predictions, val_targets),
        }

    def get_correlation_coefficient(self, dataloader):
        # TODO: Handle ties
        dataset_performances = {}
        pipeline_key = 'pipeline_ids'
        actual_key = 'f1_actuals'
        predict_key = 'f1_predictions'
        for x_batch, y_batch in dataloader:
            y_hat_batch = self.model(x_batch)

            # Get the pipeline id and the data set ids that correspond to it
            pipeline_id, pipeline, x, dataset_ids = x_batch

            # Create a list of tuples containing the pipeline id and its f1 values for each data set in this batch
            for i in range(len(dataset_ids)):
                dataset_id = dataset_ids[i]
                f1_actual = y_batch[i].item()
                f1_predict = y_hat_batch[i].item()
                if dataset_id in dataset_performances:
                    dataset_performance = dataset_performances[dataset_id]
                    pipeline_ids = dataset_performance[pipeline_key]
                    f1_actuals = dataset_performance[actual_key]
                    f1_predictions = dataset_performance[predict_key]
                    pipeline_ids.append(pipeline_id)
                    f1_actuals.append(f1_actual)
                    f1_predictions.append(f1_predict)
                else:
                    dataset_performance = {pipeline_key: [pipeline_id], actual_key: [f1_actual], predict_key: [f1_predict]}
                    dataset_performances[dataset_id] = dataset_performance

        dataset_cc_sum = 0.0
        dataset_performances = dataset_performances.values()
        for dataset_performance in dataset_performances:
            f1_actuals = dataset_performance[actual_key]
            f1_predictions = dataset_performance[predict_key]
            actual_ranks = self.rank(f1_actuals)
            predicted_ranks = self.rank(f1_predictions)

            # Get the spearman correlation coefficient for this data set
            spearman_result = scipy.stats.spearmanr(actual_ranks, predicted_ranks)
            dataset_cc = spearman_result.correlation
            dataset_cc_sum += dataset_cc
        num_datasets = len(dataset_performances)
        mean_dataset_cc = dataset_cc_sum / num_datasets
        return mean_dataset_cc

    @staticmethod
    def rank(performances):
        ranks = np.argsort(performances)[::-1]
        return ranks


def main():
    problem = Regression(seed = 0)
    print(problem.baselines)

if __name__ == '__main__':
    main()
