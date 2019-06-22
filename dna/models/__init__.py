import typing

from .dna_regression_model import DNARegressionModel
from .mean_baseline import MeanBaseline
from .median_baseline import MedianBaseline
from .per_primitive_baseline import PerPrimitiveBaseline
from .auto_sklearn_metalearner import AutoSklearnMetalearner
from .dag_rnn_regression_model import DAGRNNRegressionModel


def get_model(model_name: str, model_config: typing.Dict, seed: int):
    model_class = {
        'dna_regression': DNARegressionModel,
        'mean_regression': MeanBaseline,
        'median_regression': MedianBaseline,
        'per_primitive_regression': PerPrimitiveBaseline,
        'autosklearn': AutoSklearnMetalearner,
        'dagrnn_regression': DAGRNNRegressionModel
    }[model_name.lower()]
    init_model_config = model_config.get('__init__', {})
    return model_class(**init_model_config, seed=seed)
