import unittest

import numpy as np

from dna import utils


class UtilsTestCase(unittest.TestCase):

    def test_rank(self):
        values = [.5]
        true_rankings = [0]
        rankings = list(utils.rank(values))
        self.assertEqual(rankings, true_rankings, 'failed to rank one item')

        values = [.2, .1, .7, .6, .9]
        true_rankings = [3, 4, 1, 2, 0]
        rankings = list(utils.rank(values))
        self.assertEqual(rankings, true_rankings, 'failed to rank 5 items')