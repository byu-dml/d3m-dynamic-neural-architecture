import pandas
import numpy as np

class DNABaselines:
    def predict(self, dataset_performances_map, k=25):
        pipeline_key = 'pipeline_ids'
        actual_key = 'f1_actuals'
        predict_key = 'f1_predictions'
        dataset_cc_sum = 0.0
        dataset_performances = dataset_performances_map.values()
        top_k_out_of_total = []
        metric_differences = []
        for dataset_performance in dataset_performances:
            print("Number of pipelines for this dataset:", len(dataset_performance[actual_key]))
            f1_actuals = dataset_performance[actual_key]
            f1_predictions = dataset_performance[predict_key]
            actual_ranks = self.rank(f1_actuals)
            predicted_ranks = self.rank(f1_predictions)
            # get top k out of the total k: => do this by putting the data into a series, getting the n_largest and
            # then getting the index, which is the id
            top_k_predicted = list(
                pandas.Series(dataset_performance[predict_key], dataset_performance[pipeline_key]).nlargest(k).index)
            top_k_actual = list(
                pandas.Series(dataset_performance[actual_key], dataset_performance[pipeline_key]).nlargest(k).index)
            top_k_out_of_total.append(len(set(top_k_predicted).intersection(set(top_k_actual))))

            # get the actual values for predicted top pipeline
            best_metric_value_pred = np.nanmax(
                pandas.DataFrame(dataset_performance[predict_key], dataset_performance[pipeline_key]))
            best_metric_value = np.nanmax(
                pandas.DataFrame(dataset_performance[actual_key], dataset_performance[pipeline_key]))
            metric_differences.append(np.abs(best_metric_value_pred - best_metric_value))

            # Get the spearman correlation coefficient for this data set
            spearman_result = spearmanr(actual_ranks, predicted_ranks)
            dataset_cc = spearman_result.correlation
            dataset_cc_sum += dataset_cc

        num_datasets = len(dataset_performances)
        mean_dataset_cc = dataset_cc_sum / num_datasets
        print("On average, the top {} out of the real top {} is".format(k, k), np.mean(top_k_out_of_total))
        print("The difference in actual vs predicted is", np.mean(metric_differences))
        return mean_dataset_cc, top_k_out_of_total


    @staticmethod
    def rank(performances):
        ranks = np.argsort(performances)[::-1]
        return ranks