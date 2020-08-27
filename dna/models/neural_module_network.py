from .base_models import PyTorchRegressionRankModelBase
from .torch_modules.nmn import NMNModule


class NeuralModuleNetworkModel(PyTorchRegressionRankModelBase):

    name = 'NMN'
    color = 'fuschia'

    def __init__(
        self, n_hidden_layers: int,  hidden_layer_size: int, activation_name: str,
        use_batch_norm: bool, reduction_name: str = 'max', loss_function_name: str = 'rmse',
        use_skip: bool = False, dropout = 0.0,
        *, device='cuda:0', seed=0
    ):
        super().__init__(device=device, seed=seed, loss_function_name=loss_function_name)

        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation_name = activation_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.reduction_name = reduction_name
        self.output_layer_size = 1
        self._model_seed = self.seed + 1

    def _get_model(self, train_data):
        module_ids = self._get_module_ids(train_data)
        input_layer_size = len(train_data[0]['metafeatures'])
        return NMNModule(
            module_ids, self.n_hidden_layers + 1, input_layer_size, self.hidden_layer_size,
            self.output_layer_size, self.activation_name, self.use_batch_norm, self.use_skip,
            self.dropout, self.reduction_name, device=self.device, seed=self.seed
        )

    def _get_module_ids(self, train_data) -> set:
        module_ids = set()
        for instance in train_data:
            for step in instance['pipeline']['steps']:
                module_ids.add(step['name'])
        return module_ids
