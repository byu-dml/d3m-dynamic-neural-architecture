import unittest

import numpy as np
import pandas as pd

from dna import utils
from dna import metrics


class MetricsTestCase(unittest.TestCase):

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


class MetricsAtKTestCase(unittest.TestCase):

    def setUp(self):
        self.test_cases = [
            {
                # [3, 2, 0, 1, 3]
                # [3, 3, 2, 1, 0]
                'relevance':             [2, 3, 0, 1, 3],
                'rank':                  [1, 4, 2, 3, 0],
                'n_correct_at_k':        [1, 1, 2, 3, 5],
                'regret_at_k':           [0, 0, 0, 0, 0],
                'linear_dcg_at_k':       [3.0, 4.26185951, 4.26185951, 4.69253607, 5.85309449],
                'linear_ndcg_at_k':      [1.0, 0.87104906, 0.72323297, 0.74208293, 0.92561495],
                'exponential_dcg_at_k':  [7.0, 8.89278926, 8.89278926, 9.32346582, 12.03143547],
                'exponential_ndcg_at_k': [1.0, 0.77894125, 0.68848245, 0.69853426, 0.90142121],
            },
            {
                'relevance': [.5, 1],
                'rank': [0],
                'n_correct_at_k': [0],
                'regret_at_k': [.5],
                'linear_dcg_at_k': None,
                'linear_ndcg_at_k': None,
                'exponential_dcg_at_k': None,
                'exponential_ndcg_at_k': None,
            },
            {
                'relevance': [0, .5, 1],
                'rank': [1, 0],
                'n_correct_at_k': [0, 1],
                'regret_at_k': [.5, .5],
                'linear_dcg_at_k': None,
                'linear_ndcg_at_k': None,
                'exponential_dcg_at_k': None,
                'exponential_ndcg_at_k': None,
            },
            {
                'relevance': [0, .5, 1],
                'rank': [2, 1, 0],
                'n_correct_at_k': [1, 2, 3],
                'regret_at_k': [0, 0, 0],
                'linear_dcg_at_k': None,
                'linear_ndcg_at_k': None,
                'exponential_dcg_at_k': None,
                'exponential_ndcg_at_k': None,
            },
            {
                # [1, 0, 2, 1, 2, 3]
                # [3, 2, 2, 1, 1, 0]
                'relevance': [1, 2, 3, 0, 1, 2],
                'rank':      [0, 4, 5, 1, 3, 2],
                'n_correct_at_k': [0, 0, 1, 3, 4, 6],
                'regret_at_k': [2, 2, 1, 1, 1, 0],
                'linear_dcg_at_k': None,
                'linear_ndcg_at_k': None,
                'exponential_dcg_at_k': None,
                'exponential_ndcg_at_k': None,
            },
            # next test case taken from https://en.wikipedia.org/wiki/Discounted_cumulative_gain
            {
                'relevance': [3, 2, 3, 0, 1, 2],
                'rank':      [1, 2, 3, 4, 5, 6],
                'n_correct_at_k': None,
                'regret_at_k': None,
                'linear_dcg_at_k': [3, 4.262, 5.762, 5.762, 6.149, 6.861],
                'linear_ndcg_at_k': [1, 0.871, 0.978, 0.853, 0.861, 0.961],
                'exponential_dcg_at_k': None,
                'exponential_ndcg_at_k': None,
            },
            {
                'relevance': [3, 2, 3, 0, 1, 2],
                'rank':      [0, 2, 1, 5, 4, 3],
                'n_correct_at_k': [1, 2, 3, 4, 5, 6],
                'regret_at_k': [0, 0, 0, 0, 0, 0],
                'linear_dcg_at_k': [3.0, 4.89278926, 5.89278926, 6.75414238, 7.14099518, 7.14099518],
                'linear_ndcg_at_k': [1, 1, 1, 1, 1, 1],
                'exponential_dcg_at_k': None,
                'exponential_ndcg_at_k': None,
            }
        ]

    def _test(self, metric_at_k, key, **kwargs):
        for i, test_case in enumerate(self.test_cases):
            if test_case.get(key, None) is not None:
                try:
                    for j in range(len(test_case['rank'])):
                        k = j + 1
                        value_at_k = metric_at_k(test_case['relevance'], test_case['rank'], k, **kwargs)
                        np.testing.assert_almost_equal(
                            test_case[key][j], value_at_k, decimal=3,
                            err_msg='{} failed on test {} with k={}. true value: {}; computed value: {}.'.format(
                                key, i, k, test_case[key][j], value_at_k
                            )
                        )
                    values_at_k = metric_at_k(test_case['relevance'], test_case['rank'], **kwargs)
                    np.testing.assert_almost_equal(
                        test_case[key], values_at_k, decimal=3,
                        err_msg='{} failed on test {} over all k. true value: {}; computed value: {}.'.format(
                            key, i, test_case[key], values_at_k
                        )
                    )
                except IndexError as e:
                    self.fail('{} erred on test case {} with\n{}'.format(key, i, e))

    def test_n_correct_at_k(self):
        self._test(metrics.n_correct_at_k, 'n_correct_at_k')

    def test_regreat_at_k(self):
        self._test(metrics.regret_at_k, 'regret_at_k')

    def test_linear_dcg_at_k(self):
        self._test(metrics.dcg_at_k, 'linear_dcg_at_k', gains_f='linear')

    def test_exponential_dcg_at_k(self):
        self._test(metrics.dcg_at_k, 'exponential_dcg_at_k', gains_f='exponential')

    def test_linear_ndcg_at_k(self):
        self._test(metrics.ndcg_at_k, 'linear_ndcg_at_k', gains_f='linear')

    def test_exponential_ndcg_at_k(self):
        self._test(metrics.ndcg_at_k, 'exponential_ndcg_at_k', gains_f='exponential')
