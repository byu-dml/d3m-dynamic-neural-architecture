from .base_models import PyTorchRegressionRankModelBase


class NeuralModuleNetwork(PyTorchRegressionRankModelBase):

    name = 'NMN'
    color = 'fuschia'

    def __init__(self, device='cuda:0', seed=0):
        super().__init__(device, seed)

    def _get_model(self, train_data):
        pass
