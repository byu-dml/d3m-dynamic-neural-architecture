import unittest

from dna.metrics import ndcg_at_k, spearman_correlation
from dna.problems import RankProblem
from dna import utils


class RankProblemTestCase(unittest.TestCase):

    def test_score(self):
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
        dataset_1_test_f1_macro = [targets[i]['test_f1_macro'] for i in range(3)]
        dataset_2_test_f1_macro = [targets[i]['test_f1_macro'] for i in range(3,6)]
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
        dataset_1_predicted_rank = predictions['dataset_1']['rank']
        dataset_2_predicted_rank = predictions['dataset_2']['rank']

        problem_scores = problem.score(predictions, targets)

        dataset_1_spearman = spearman_correlation(dataset_1_predicted_rank, utils.rank(dataset_1_test_f1_macro))
        self.assertEqual(dataset_1_spearman[0], problem_scores['per_dataset_scores']['dataset_1']['spearman_correlation']['correlation_coefficient'])
        self.assertEqual(dataset_1_spearman[1], problem_scores['per_dataset_scores']['dataset_1']['spearman_correlation']['p_value'])

        dataset_2_spearman = spearman_correlation(dataset_2_predicted_rank, utils.rank(dataset_2_test_f1_macro))
        self.assertEqual(dataset_2_spearman[0], problem_scores['per_dataset_scores']['dataset_2']['spearman_correlation']['correlation_coefficient'])
        self.assertEqual(dataset_2_spearman[1], problem_scores['per_dataset_scores']['dataset_2']['spearman_correlation']['p_value'])

        dataset_1_ndcg = ndcg_at_k(dataset_1_test_f1_macro, dataset_1_predicted_rank)
        self.assertEqual(dataset_1_ndcg, problem_scores['per_dataset_scores']['dataset_1']['ndcg'])

        dataset_2_ndcg = ndcg_at_k(dataset_2_test_f1_macro, dataset_2_predicted_rank)
        self.assertEqual(dataset_2_ndcg, problem_scores['per_dataset_scores']['dataset_2']['ndcg'])
