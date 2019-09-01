import typing
import unittest

from dna.metrics import n_correct_at_k, ndcg_at_k, regret_at_k, spearman_correlation
from dna.problems import RankProblem
from dna import utils


class RankProblemScoreTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.problem = RankProblem('dataset_id')
        targets = [
            {
                'dataset_id': 'dataset_1',
                'pipeline_id': 'a',
                'test_f1_macro': 0.9,
            },
            {
                'dataset_id': 'dataset_1',
                'pipeline_id': 'b',
                'test_f1_macro': 0.8,
            },
            {
                'dataset_id': 'dataset_1',
                'pipeline_id': 'c',
                'test_f1_macro': 0.7,
            },
            {
                'dataset_id': 'dataset_2',
                'pipeline_id': 'a',
                'test_f1_macro': 0.7,
            },
            {
                'dataset_id': 'dataset_2',
                'pipeline_id': 'b',
                'test_f1_macro': 0.6,
            },
            {
                'dataset_id': 'dataset_2',
                'pipeline_id': 'c',
                'test_f1_macro': 0.5,
            },
        ]
        cls.dataset_1_test_f1_macro = [targets[i]['test_f1_macro'] for i in range(3)]
        cls.dataset_2_test_f1_macro = [targets[i]['test_f1_macro'] for i in range(3,6)]
        predictions = {
            'dataset_1': {
                'pipeline_id': [
                    'a',
                    'b',
                    'c',
                ],
                'rank': [
                    0,
                    1,
                    2,
                ]
            },
            'dataset_2': {
                'pipeline_id': [
                    'a',
                    'b',
                    'c',
                ],
                'rank': [
                    2,
                    1,
                    0,
                ]
            }
        }
        cls.dataset_1_predicted_rank = predictions['dataset_1']['rank']
        cls.dataset_2_predicted_rank = predictions['dataset_2']['rank']

        cls.problem_scores = cls.problem.score(predictions, targets)

    def test_spearman_correlation(self):
        dataset_1_spearman = spearman_correlation(self.dataset_1_predicted_rank, utils.rank(self.dataset_1_test_f1_macro))
        self.assertEqual(dataset_1_spearman[0], self.problem_scores[self.problem.group_scores_key]['dataset_1']['spearman_correlation'])
        self.assertEqual(dataset_1_spearman[1], self.problem_scores[self.problem.group_scores_key]['dataset_1']['spearman_p_value'])

        dataset_2_spearman = spearman_correlation(self.dataset_2_predicted_rank, utils.rank(self.dataset_2_test_f1_macro))
        self.assertEqual(dataset_2_spearman[0], self.problem_scores[self.problem.group_scores_key]['dataset_2']['spearman_correlation'])
        self.assertEqual(dataset_2_spearman[1], self.problem_scores[self.problem.group_scores_key]['dataset_2']['spearman_p_value'])

        mean_spearman = (dataset_1_spearman[0] + dataset_2_spearman[0]) / 2
        self.assertEqual(mean_spearman, self.problem_scores[self.problem.agg_scores_key]['spearman_correlation_mean'])

    def _test_metric_at_k(self, metric_at_k: typing.Callable):
        metric_name = metric_at_k.__name__
        all_values_at_k = []
        for k in range(1,4):
            dataset_1_value = metric_at_k(self.dataset_1_test_f1_macro, self.dataset_1_predicted_rank, k)
            self.assertEqual(dataset_1_value, self.problem_scores[self.problem.group_scores_key]['dataset_1'][metric_name][k-1])
            all_values_at_k.append(dataset_1_value)

            dataset_2_value = metric_at_k(self.dataset_2_test_f1_macro, self.dataset_2_predicted_rank, k)
            self.assertEqual(dataset_2_value, self.problem_scores[self.problem.group_scores_key]['dataset_2'][metric_name][k-1])
            all_values_at_k.append(dataset_2_value)

            values_at_k_mean = (dataset_1_value + dataset_2_value) / 2
            self.assertEqual(values_at_k_mean, self.problem_scores[self.problem.agg_scores_key]['{}_mean'.format(metric_name)][k-1])

        values_sum = sum(all_values_at_k) / len(all_values_at_k)
        self.assertEqual(values_sum, self.problem_scores[self.problem.total_scores_key]['{}_mean'.format(metric_name)])

    def test_ndcg(self):
        self._test_metric_at_k(ndcg_at_k)

    def test_regret_at_k(self):
        self._test_metric_at_k(regret_at_k)

    def test_n_correct_at_k(self):
        self._test_metric_at_k(n_correct_at_k)
