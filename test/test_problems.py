import unittest

from dna.metrics import ndcg_at_k, spearman_correlation, top_k_regret, top_k_correct
from dna.problems import RankProblem
from dna import utils


class RankProblemScoreTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        problem = RankProblem('dataset_id')
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

        cls.problem_scores = problem.score(predictions, targets)

    def test_spearman_correlation(self):
        dataset_1_spearman = spearman_correlation(self.dataset_1_predicted_rank, utils.rank(self.dataset_1_test_f1_macro))
        self.assertEqual(dataset_1_spearman[0], self.problem_scores['per_dataset_scores']['dataset_1']['spearman_correlation']['correlation_coefficient'])
        self.assertEqual(dataset_1_spearman[1], self.problem_scores['per_dataset_scores']['dataset_1']['spearman_correlation']['p_value'])

        dataset_2_spearman = spearman_correlation(self.dataset_2_predicted_rank, utils.rank(self.dataset_2_test_f1_macro))
        self.assertEqual(dataset_2_spearman[0], self.problem_scores['per_dataset_scores']['dataset_2']['spearman_correlation']['correlation_coefficient'])
        self.assertEqual(dataset_2_spearman[1], self.problem_scores['per_dataset_scores']['dataset_2']['spearman_correlation']['p_value'])

        mean_spearman = (dataset_1_spearman[0] + dataset_2_spearman[0]) / 2
        self.assertEqual(mean_spearman, self.problem_scores['mean_scores']['spearman_correlation']['mean'])

    def test_ndcg(self):
        ndcgs = []
        for k in range(1,4):
            dataset_1_ndcg = ndcg_at_k(self.dataset_1_test_f1_macro, self.dataset_1_predicted_rank, k)
            self.assertEqual(dataset_1_ndcg, self.problem_scores['per_dataset_scores']['dataset_1']['ndcg_over_k'][k-1])
            ndcgs.append(dataset_1_ndcg)

            dataset_2_ndcg = ndcg_at_k(self.dataset_2_test_f1_macro, self.dataset_2_predicted_rank, k)
            self.assertEqual(dataset_2_ndcg, self.problem_scores['per_dataset_scores']['dataset_2']['ndcg_over_k'][k-1])
            ndcgs.append(dataset_2_ndcg)

            mean_ndcg_over_k = (dataset_1_ndcg + dataset_2_ndcg) / 2
            self.assertEqual(mean_ndcg_over_k, self.problem_scores['mean_scores']['ndcg_over_k'][k-1])

        mean_ndcg = sum(ndcgs) / len(ndcgs)
        self.assertEqual(mean_ndcg, self.problem_scores['mean_scores']['mean_ndcg'])

    def test_top_k_regret(self):
        for k in range(1,4):
            regret_1 = top_k_regret(self.dataset_1_test_f1_macro, self.dataset_1_predicted_rank, k)
            self.assertEqual(regret_1, self.problem_scores['per_dataset_scores']['dataset_1']['top_k_regrets'][k-1])

            regret_2 = top_k_regret(self.dataset_2_test_f1_macro, self.dataset_2_predicted_rank, k)
            self.assertEqual(regret_2, self.problem_scores['per_dataset_scores']['dataset_2']['top_k_regrets'][k-1])

            mean_top_k_regret = (regret_1 + regret_2) / 2
            self.assertEqual(mean_top_k_regret, self.problem_scores['mean_scores']['mean_top_k_regrets'][k-1])

    def test_top_k_count(self):
        for k in range(1,4):
            count_1 = top_k_correct(self.dataset_1_test_f1_macro, self.dataset_1_predicted_rank, k)
            self.assertEqual(count_1, self.problem_scores['per_dataset_scores']['dataset_1']['top_k_counts'][k-1])

            count_2 = top_k_correct(self.dataset_2_test_f1_macro, self.dataset_2_predicted_rank, k)
            self.assertEqual(count_2, self.problem_scores['per_dataset_scores']['dataset_2']['top_k_counts'][k-1])

            mean_count = (count_1 + count_2) / 2
            self.assertEqual(mean_count, self.problem_scores['mean_scores']['mean_top_k_counts'][k-1])
