from .models import PyTorchRegressionRankSubsetModelBase
from .torch_modules.dag_transformer.dag_transformer import DAGTransformer


class DAGTransformerRegressionModel(PyTorchRegressionRankSubsetModelBase):
    def _get_model(self, train_data):
        return DAGTransformer()

    def _get_loss_function(self):
        pass

    def _get_optimizer(self, learning_rate):
        pass

    def _get_data_loader(self, data, batch_size, drop_last, shuffle):
        pass