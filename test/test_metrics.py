import unittest

import numpy as np
import pandas as pd

from dna import utils
from dna import metrics


class MetricsTestCase(unittest.TestCase):

    def format_and_get_top_k(self, predicted_top_k, target_ids, target_scores, k, top_k_function):
        """
        :param predicted_top_k: the top-k ids
        :param target_ids: a list of the pipeline ids
        :param target_scores: a list of the pipeline scores
        :param k: the number of pipelines to get
        :param top_k_function: the top k function to evaluate, top_k_regret or top_k_correct
        """
        targets = pd.DataFrame({'pipeline_id': target_ids, 'test_f1_macro': target_scores})
        return top_k_function(predicted_top_k, targets, k)

    def format_and_get_spearman(self, actual_data, ranked_data):
        """
        :param actual_data: a list of the real rankings
        :param ranked_data: a list of the ranked predictions
        """
        actual_data, ranked_data = self.format_for_rank(actual_data, ranked_data)
        metric = metrics.spearman_correlation(ranked_data, actual_data)
        return metric

    def format_for_rank(self, actual_data, ranked_data):
        actual_data = pd.DataFrame({'test_f1_macro': actual_data})
        ranked_data = pd.DataFrame({'rank': ranked_data})
        return actual_data, ranked_data

    def test_rmse(self):
        values_pred = [.5]
        values_actual = [0]
        true_value = (.5**2)**.5
        metric = metrics.rmse(values_pred, values_actual)
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get rmse of one item')

        values_pred = [.1, .2, .3, .4, .5]
        values_actual = [.1, .2, .3, .4, .5]
        true_value = 0
        metric = metrics.rmse(values_pred, values_actual)
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get rmse of perfect')

        values_pred = [.1, .2, .3, .4, .5]
        values_actual = [.2, .3, .4, .5, .6]
        true_value = (.1**2)**.5
        metric = metrics.rmse(values_pred, values_actual)
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get rmse of mixed values')

    def test_accuracy(self):
        values_pred = [1]
        values_actual = [0]
        true_value = 0
        metric = metrics.accuracy(values_pred, values_actual)
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get accuracy of one item')

        values_pred = [1, 1, 1, 1, 1]
        values_actual = [1, 1, 1, 1, 1]
        true_value = 1
        metric = metrics.accuracy(values_pred, values_actual)
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get accuracy of perfect')

        values_pred = [1, 1, 1, 1, 1]
        values_actual = [0, 1, 0, 1, 1]
        true_value = .6
        metric = metrics.accuracy(values_pred, values_actual)
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get accuracy of mixed values')

    def test_spearman(self):
        """
        Example #1 from SciPy at https://github.com/scipy/scipy/blob/master/scipy/stats/stats.py line 3669
        Example #2 from https://chrisalbon.com/statistics/frequentist/spearmans_rank_correlation/

        We have to pass the values to spearman and it will rank them OR we pass the ranked values where #1 is the lowest
        """
        metric = self.format_and_get_spearman([1,2,3,4,5], [5,6,7,8,7])
        true_metric = (0.82078268166812329, 0.088587005313543798)
        np.testing.assert_almost_equal(
            metric, true_metric, err_msg='failed to get spearman from scipy, was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

        metric = self.format_and_get_spearman([1,2,3,4,5,6,7,8,9], [2,1,2,4.5,7,6.5,6,9,9.5])
        true_metric = 0.90377360145618091
        np.testing.assert_almost_equal(
            metric[0], true_metric,
            err_msg='failed to get spearman from second example, was {}, shouldve been {}'.format(metric, true_metric)
        )

        metric = self.format_and_get_spearman([1,2,3], [2,4,6])
        true_metric = (1.0, 0)
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get spearman from perfect example, was {}, shouldve been {}'.format(metric, true_metric)
        )

        metric = self.format_and_get_spearman([3, 2, 1],[2, 4, 6] )
        true_metric = (-1.0, 0)
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get spearman from inverse example, was {}, shouldve been {}'.format(metric, true_metric)
        )

        random_n = 50000
        metric = self.format_and_get_spearman(list(np.random.rand(random_n)),list(np.random.rand(random_n)))
        true_metric = 0
        np.testing.assert_almost_equal(
            metric[0], true_metric, decimal=2,
            err_msg='failed to get spearman from random example, was {}, shouldve been {}'.format(metric, true_metric)
        )

        metric = self.format_and_get_spearman([1, 2.5, 2.5],[1, 2, 3] )
        true_metric = (0.866025403784, 0.333333333333)
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get spearman from tie example, was {}, shouldve been {}'.format(metric, true_metric)
        )

    def test_dcg(self):
        """
        Examples from:
        https://gist.github.com/mblondel/7337391
        https://machinelearningmedium.com/2017/07/24/discounted-cumulative-gain/
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
        """
        # Check that some rankings are better than others
        assert metrics.dcg_at_k([5, 3, 2], [2, 1, 0]) > metrics.dcg_at_k([4, 3, 2], [2, 1, 0])
        assert metrics.dcg_at_k([4, 3, 2], [2, 1, 0]) > metrics.dcg_at_k([1, 3, 2], [2, 1, 0])
        assert metrics.dcg_at_k([5, 3, 2], [2, 1, 0], k=2) == metrics.dcg_at_k([4, 3, 2], [2, 1, 0], k=2)
        assert metrics.dcg_at_k([4, 3, 2], [2, 1, 0], k=2) == metrics.dcg_at_k([1, 3, 2], [2, 1, 0], k=2)

        # Check that sample order is irrelevant
        assert metrics.dcg_at_k([5, 3, 2], [2, 1, 0]) == metrics.dcg_at_k([2, 3, 5], [0, 1, 2])
        assert metrics.dcg_at_k([5, 3, 2], [2, 1, 0], k=2) == metrics.dcg_at_k([2, 3, 5], [0, 1, 2], k=2)

        # wikipedia example
        true_metric = 6.86112668859
        ground_truth = [3, 2, 3, 0, 1, 2]
        predictions = [1, 2, 3, 4, 5, 6]
        metric = metrics.dcg_at_k(ground_truth, predictions, k=6, gains_f='linear')
        np.testing.assert_almost_equal(
            metric, true_metric, err_msg='failed to get the correct ndcg, was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

    # TODO: test exponential (n)dcg

    def test_ndcg(self):
        """
        Examples from:
        https://gist.github.com/mblondel/7337391
        https://machinelearningmedium.com/2017/07/24/discounted-cumulative-gain/
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
        """
        # Perfect rankings
        assert metrics.ndcg_at_k([5, 3, 2], [0, 1, 2]) == 1.0
        assert metrics.ndcg_at_k([2, 3, 5], [2, 1, 0]) == 1.0
        assert metrics.ndcg_at_k([5, 3, 2], [0, 1, 2], k=2) == 1.0
        assert metrics.ndcg_at_k([2, 3, 5], [2, 1, 0], k=2) == 1.0

        # wikipedia example
        true_metric = 0.96080819
        ground_truth = [3, 2, 3, 0, 1, 2]
        predictions = [1, 2, 3, 4, 5, 6]
        metric = metrics.ndcg_at_k(ground_truth, predictions, k=6, gains_f='linear')
        np.testing.assert_almost_equal(
            metric, true_metric, err_msg='failed to get the correct ndcg, was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

    def test_pearson_correlation(self):
        """
        Examples #4 from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
        Since our code is literally the same as theirs, this is just a sanity check
        """
        values_pred = [.3, .4, .5]
        values_actual = [.3, .4, .5]
        true_value = 1
        metric = metrics.pearson_correlation(values_pred, values_actual)
        np.testing.assert_almost_equal(metric[0], true_value, err_msg='failed to get Pearson of perfect')

        values_pred = [.5, .4, .3]
        values_actual = [.3, .4, .5]
        true_value = -1
        metric = metrics.pearson_correlation(values_pred, values_actual)
        np.testing.assert_almost_equal(metric[0], true_value, err_msg='failed to get Pearson of inverse')

        values_pred = [5., .4, .3]
        values_actual = [1, 1, 1]
        true_value = np.nan  # in the limit defined as nan
        metric = metrics.pearson_correlation(values_pred, values_actual)
        np.testing.assert_almost_equal(metric[0], true_value, err_msg='failed to get Pearson of NaN')

        true_value = (-0.7426106572325057, 0.1505558088534455)
        metric = metrics.pearson_correlation( [1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
        np.testing.assert_almost_equal(metric, true_value, err_msg='failed to get Pearson of NaN')

    def test_top_k_correct(self):
        k = 1
        metric = metrics.top_k_correct([.5, 1], [0], k)
        true_metric = 0
        np.testing.assert_almost_equal(
            metric, true_metric, err_msg='failed to get top_1 ranking from 0 example: was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

        k = 2
        metric = metrics.top_k_correct([0, .5, 1], [1, 0], k)
        true_metric = 1
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get top_1 ranking from perfect example: was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

        k = 3
        metric = metrics.top_k_correct([.5, 1, 0], [1, 0, 2], k)
        true_metric = 3
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get top_3 ranking from perfect example: was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

    def test_top_k_regret(self):
        k = 1
        metric = metrics.top_k_regret([.5, 1], [0], k)
        true_metric = .5
        np.testing.assert_almost_equal(
            metric, true_metric, err_msg='failed to get top_1 ranking from 0 example: was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

        k=2
        metric = metrics.top_k_regret([.5, 1, 0], [1, 0], k)
        true_metric = 0
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get top_1 ranking from perfect example: was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )

        k = 3
        metric = metrics.top_k_regret([.5, 1, 0], [1, 0, 2], k)
        true_metric = 0
        np.testing.assert_almost_equal(
            metric, true_metric,
            err_msg='failed to get top_3 ranking from perfect example: was {}, shouldve been {}'.format(
                metric, true_metric
            )
        )
