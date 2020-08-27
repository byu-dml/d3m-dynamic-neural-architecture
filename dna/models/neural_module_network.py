from .base_models import PyTorchRegressionRankModelBase
from .torch_modules.nmn import NMNModule


class NeuralModuleNetworkModel(PyTorchRegressionRankModelBase):

    name = 'NMN'
    color = 'fuschia'

    def __init__(self, device='cuda:0', seed=0):
        super().__init__(device, seed)

    def _get_model(self, train_data):
        module_ids = self._get_module_ids(train_data)
        return NMNDAGLSTMMLP(module_ids)

    def _get_module_ids(self, train_data) -> set:
        module_ids = set()
        for instance in train_data:
            for step in instance['pipeline']['steps']:
                module_ids.add(step['name'])
        return module_ids
