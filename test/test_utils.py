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

    def test_flatten(self):
        to_flatten = {
            'a': {
                'b': {
                    'c': {
                        'd': 0
                    },
                    'e': {
                        'f': 'hello'
                    },
                    'g': {
                        'h': 'world'
                    },
                    'i': {
                        'j': -3.14
                    },
                }
            }
        }
        expected_flattened = {
            'a.b.c.d': 0,
            'a.b.e.f': 'hello',
            'a.b.g.h': 'world',
            'a.b.i.j': -3.14,
        }
        flattened = utils.flatten(to_flatten)
        self.assertEqual(expected_flattened, flattened)

    def transpose_jagged_2darray(self):
        jagged_2darray = [
            [0, 1],
            [2, 3, 4],
            [5],
            [6, 7, 8, 9],
        ]
        desired_transpose = {
            0: [0, 2, 5, 6],
            1: [1, 3, 7],
            2: [4, 8],
            3: [9],
        }
        transpose = utils.transpose_jagged_2darray(jagged_2darray)
        self.assertEqual(desired_transpose, transpose)
