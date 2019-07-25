import torch

from .torch_modules.dna_module import DNAModule
from .base_models import PyTorchRegressionRankSubsetModelBase
from dna.data import GroupDataLoader, Dataset

class DNARegressionModel(PyTorchRegressionRankSubsetModelBase):

    def __init__(
            self, n_hidden_layers: int, hidden_layer_size: int, activation_name: str, use_batch_norm: bool,
            use_skip: bool = False, dropout = 0.0, *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__(y_dtype=torch.float32, device=device, seed=seed)

        self.n_hidden_layers = n_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.activation_name = activation_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.output_layer_size = 1
        self._model_seed = self.seed + 1

    def _get_model(self, train_data):
        submodule_input_sizes = {}
        for instance in train_data:
            for step in instance['pipeline']['steps']:
                submodule_input_sizes[step['name']] = len(step['inputs'])
        self.input_layer_size = len(train_data[0]['metafeatures'])

        return DNAModule(
            submodule_input_sizes, self.n_hidden_layers + 1, self.input_layer_size, self.hidden_layer_size,
            self.output_layer_size, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout,
            device=self.device, seed=self._model_seed
        )

    def _get_loss_function(self):
        objective = torch.nn.MSELoss(reduction='mean')
        return lambda y_hat, y: torch.sqrt(objective(y_hat, y))

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last, shuffle=True):
        return GroupDataLoader(
            data = data,
            group_key = 'pipeline.id',
            dataset_class = Dataset,
            dataset_params = {
                'features_key': 'metafeatures',
                'target_key': 'test_f1_macro',
                'y_dtype': self.y_dtype,
                'device': self.device
            },
            batch_size = batch_size,
            drop_last = drop_last,
            shuffle = shuffle,
            seed = self.seed + 2
        )
