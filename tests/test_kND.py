from unittest import TestCase
from pandas import DataFrame
from pandas import Series
from numpy import array

from dna.kND import KNearestDatasets


class KNearestDatasetsTestCase(TestCase):

    def test_knn_regression(self):
        d1 = 'dataset_1'
        mfs1 = [1.0, 1.0, 1.0]
        d2 = 'dataset_2'
        mfs2 = [2.0, 2.0, 2.0]
        p1 = 'pipeline_1'
        p2 = 'pipeline_2'
        p3 = 'pipeline_3'

        d1p1 = 0.9
        d1p2 = 0.8
        d1p3 = 0.7
        d2p1 = 0.7
        d2p2 = 0.6
        d2p3 = 0.5

        all_mfs = DataFrame([mfs1, mfs2], [d1, d2])
        runs = DataFrame({d1: {p1: d1p1, p2: d1p2, p3: d1p3}, d2: {p1: d2p1, p2: d2p2, p3: d2p3}})
        model = KNearestDatasets()
        model.fit(all_mfs, runs)

        # Test on metafeatures of a new data set
        regressed_values_indices = [p1, p2, p3]
        mfs = Series([2.0, 2.0, 4.0])
        actuals: Series = model.knn_regression(mfs)
        expected = Series(array([0.53, 0.46, 0.39]) / 0.7, regressed_values_indices)
        self.assertTrue(actuals.equals(expected))

        # Test on in training set metafeatures
        mfs = Series(mfs1)
        actuals = model.knn_regression(mfs)
        expected = Series([d1p1, d1p2, d1p3], regressed_values_indices)
        self.assertTrue(actuals.equals(expected))

        mfs = Series(mfs2)
        actuals = model.knn_regression(mfs)
        expected = Series([d2p1, d2p2, d2p3], regressed_values_indices)
        self.assertTrue(actuals.equals(expected))
