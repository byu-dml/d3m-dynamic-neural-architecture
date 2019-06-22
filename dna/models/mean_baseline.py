from .base_models import RegressionModelBase
from .models import ModelNotFitError


class MeanBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        RegressionModelBase.__init__(self, seed=seed)
        self.mean = None

    def fit(self, data, *, validation_data=None, output_dir=None, verbose=False):
        total = 0
        for instance in data:
            total += instance['test_f1_macro']
        self.mean = total / len(data)
        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.mean is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.mean] * len(data)


