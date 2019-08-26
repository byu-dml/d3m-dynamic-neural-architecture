import unittest

import torch

from dna.models.torch_modules import get_reduction


class TorchUtilsGetReductionTestCase(unittest.TestCase):

    def setUp(self):
        self.tensor = torch.randn(3,5)

    def _assert_tensors_equal(self, a: torch.Tensor, b: torch.Tensor):
        self.assertTrue(torch.all(torch.eq(a, b)))

    def test_mean(self):
        reduction = get_reduction('mean')
        self.assertEqual(reduction, torch.mean)

    def test_sum(self):
        reduction = get_reduction('sum')
        self.assertEqual(reduction, torch.sum)

    def test_prod(self):
        reduction = get_reduction('prod')
        self.assertEqual(reduction, torch.prod)

    def test_max(self):
        reduction = get_reduction('max')
        self._assert_tensors_equal(reduction(self.tensor, dim=0), torch.max(self.tensor, dim=0).values)

    def test_median(self):
        reduction = get_reduction('median')
        self._assert_tensors_equal(reduction(self.tensor, dim=0), torch.median(self.tensor, dim=0).values)
