from .base_models import RNNRegressionRankSubsetModelBase
from .torch_modules.dag_transformer.dag_transformer import DAGTransformer


class DAGTransformerRegressionModel(RNNRegressionRankSubsetModelBase):
    def __init__(self, device: str = 'cuda:0', seed: int = 0):

        super().__init__(device=device, seed=seed)

    def _get_model(self, train_data):
        return DAGTransformer()

    def fit(self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
            verbose=False):

        # TODO: Create attention pipeline structures

        super().fit(
            train_data, n_epochs, learning_rate, batch_size, drop_last, validation_data=validation_data,
            output_dir=output_dir, verbose=verbose
        )

