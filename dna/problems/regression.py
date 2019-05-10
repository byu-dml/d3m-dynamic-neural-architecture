import torch
import numpy as np

from data import TRAIN_DATA_PATH, TEST_DATA_PATH
from .base_problem import BaseProblem
from .regression_data_loader import RegressionDataLoader
from models import PrimitiveModel, RegressionModel, DNAModel
from scipy.stats import spearmanr

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

class Regression(BaseProblem):

    def __init__(
        self, train_data_path: str = TRAIN_DATA_PATH,
        test_data_path: str = TEST_DATA_PATH, n_folds: int = 5,
        batch_size = 32, drop_last = False, device = "cuda:0", seed = 0
    ):
        objective = torch.nn.MSELoss(reduction="mean")
        self._loss_function = lambda y, y_hat: torch.sqrt(objective(y, y_hat))
        super(Regression, self).__init__(
            train_data_path = train_data_path,
            test_data_path = test_data_path,
            n_folds = n_folds,
            batch_size = batch_size,
            drop_last = drop_last,
            device = device,
            seed = seed,
        )
        self._shape = (len(self._train_data[0][self.features_key]), 1)
        self._init_model()

    def _init_model(self):
        torch_state = torch.random.get_rng_state()

        torch.manual_seed(self._randint())
        torch.cuda.manual_seed_all(self._randint())
        input_model = PrimitiveModel("input", self._shape[0], self._shape[0])
        input_model.cuda()
        submodels = {}
        for item in self._train_data:
            primitive_names = [dict_obj["name"] for dict_obj in item[self.pipeline_key]]
            for primitive_name in primitive_names:
                if not primitive_name in submodels:
                    try:
                        n_inputs = lookup_input_size[primitive_name]
                    except KeyError as e:
                        n_inputs = 1
                    submodels[primitive_name] = PrimitiveModel(
                        primitive_name, n_inputs * self._shape[0], self._shape[0]
                    )
                    submodels[primitive_name].cuda()
        output_model = RegressionModel(self._shape[0])
        output_model.cuda()
        self._model = DNAModel(input_model, submodels, output_model)
        if "cuda" in self.device:
            self._model.cuda()
        torch.random.set_rng_state(torch_state)

    def _get_data_loader(self, data):
        return RegressionDataLoader(
            data = data,
            group_key = self.batch_group_key,
            pipeline_key = self.pipeline_key,
            dataset_params = {
                'data_set_key': self.data_set_key,
                "features_key": self.features_key,
                "target_key": self.target_key,
                "device": self.device
            },
            batch_size = self.batch_size,
            drop_last = self.drop_last,
            shuffle = True,
            seed = self._randint()
        )

    def _compute_baselines(self):
        primitive_scores = {}

        # For each pipeline
        for item in self.train_data:
            f1 = item[self.target_key]
            pipeline = item[self.pipeline_key]
            for primitive in pipeline:
                # Append the f1 value to the list of f1 values of this pipeline dataset pair for this primitive
                primitive = primitive['name']
                if primitive in primitive_scores:
                    score_list = primitive_scores[primitive]
                    score_list.append(f1)
                else:
                    primitive_scores[primitive] = [f1]

        # Compute the average f1 value of the f1 list for each primitive
        for primitive in primitive_scores:
            score_list = primitive_scores[primitive]
            primitive_score = np.mean(score_list)
            primitive_scores[primitive] = primitive_score.item()

        # Compute training and validation RMSE using this naive model
        training_RMSE = self.evaluate_baseline(primitive_scores, self.train_data)
        validation_RMSE = self.evaluate_baseline(primitive_scores, self.validation_data)
        print('Baseline Training RMSE:', training_RMSE)
        print('Baseline Validation RMSE:', validation_RMSE)

    def evaluate_baseline(self, primitive_scores, dataset):
        SE = 0.0
        for item in dataset:
            pipeline = item[self.pipeline_key]
            f1_sum = 0.0
            for primitive in pipeline:
                primitive = primitive['name']
                f1 = primitive_scores[primitive]
                f1_sum += f1
            f1_predict = f1_sum / len(pipeline)
            f1_actual = item[self.target_key]
            SE += (f1_actual - f1_predict) ** 2
        MSE = SE / len(dataset)
        RMSE = np.sqrt(MSE)
        return RMSE

    def _compute_mean_baseline(self):
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
            "min_mean_med_rmse": {
                "train": train_rmse,
                "validation": validation_rmse
            }
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
            spearman_result = spearmanr(actual_ranks, predicted_ranks)
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
