import typing

import torch.nn as nn

from . import torch_utils


class NeuralModuleNetwork(nn.Module):
    """
    A dynamic "Neural Module Network", patterned after

    Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. "Neural module networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 39-48. 2016.

    This is the core dynamic component that constructs the DAG network dynamically at runtime based on each instance of data.
    """


    class NMNModule(nn.Module):
        """
        An individual module used by the Neural Module Network
        """

        def __init__(self, input_reduction_name: str = None):
            super().__init__()
            self.input_reduction_name = input_reduction_name

        def forward(self, x):
            pass


    def __init__(self, modules: nn.ModuleDict, *, device: str = None, seed: int = 0):
        super().__init__()
        self.modules = modules

    def _get_forward_network(self) -> nn.Module:
        pass

    def forward(self, x):
        pass


class NMNDAGLSTMMLP(nn.Module):

    def __init__(self, module_ids: typing.Sequence):
        super().__init__()

        modules = self._init_modules(module_ids)
        self._nmn = NeuralModuleNetwork(modules)
        # self._daglstm = DAGLSTM()

    def _init_modules(self, module_ids: typing.Sequence) -> nn.ModuleDict:
        modules = nn.ModuleDict()
        for id_ in module_ids:
            modules[id_] = NeuralModuleNetwork.NMNModule()
        return modules
