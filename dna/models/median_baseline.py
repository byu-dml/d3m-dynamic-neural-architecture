import numpy as np

from .base_models import RegressionModelBase
from .models import ModelNotFitError


class MedianBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        RegressionModelBase.__init__(self, seed=seed)
        self.median = None

    def fit(self, data, *, validation_data=None, output_dir=None, verbose=False):
        self.median = np.median([instance['test_f1_macro'] for instance in data])
        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.median is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.median] * len(data)


