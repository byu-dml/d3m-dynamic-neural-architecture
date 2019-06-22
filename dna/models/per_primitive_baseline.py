from .base_models import RegressionModelBase
from .models import ModelNotFitError


class PerPrimitiveBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        RegressionModelBase.__init__(self, seed=seed)
        self.primitive_scores = None

    def fit(self, data, *, validation_data=None, output_dir=None, verbose=False):
        # for each primitive, get the scores of all the pipelines that use the primitive
        primitive_score_totals = {}
        for instance in data:
            for primitive in instance['pipeline']['steps']:
                if primitive['name'] not in primitive_score_totals:
                    primitive_score_totals[primitive['name']] = {
                        'total': 0,
                        'count': 0,
                    }
                primitive_score_totals[primitive['name']]['total'] += instance['test_f1_macro']
                primitive_score_totals[primitive['name']]['count'] += 1

        # compute the average pipeline score per primitive
        self.primitive_scores = {}
        for primitive_name in primitive_score_totals:
            total = primitive_score_totals[primitive_name]['total']
            count = primitive_score_totals[primitive_name]['count']
            self.primitive_scores[primitive_name] = total / count

        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.primitive_scores is None:
            raise ModelNotFitError('PerPrimitiveBaseline not fit')

        predictions = []
        for instance in data:
            prediction = 0
            for primitive in instance['pipeline']['steps']:
                prediction += self.primitive_scores[primitive['name']]
            prediction /= len(instance['pipeline']['steps'])
            predictions.append(prediction)

        return predictions


