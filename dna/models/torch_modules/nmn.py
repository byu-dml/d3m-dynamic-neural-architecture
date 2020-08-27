import typing

import torch
import torch.nn as nn

from .dag import DAGModule
from .daglstm import DAGLSTMModule
from .fully_connected import FullyConnectedModule
from .pipeline_dag_builder import PipelineDAGBuilder
from .torch_utils import get_activation, get_reduction


class NMNModule(nn.Module):
    """
    A Neural Module Network, modified to support DAG-structured data with a DAGLSTM instead of a
    vanilla LSTM. See:

    Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. "Neural module networks." In
    Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 39-48. 2016.
    """

    def __init__(
        self, module_ids: typing.Sequence[str], n_layers: int, input_layer_size: int,
        hidden_layer_size: int, output_layer_size: int, activation_name: str, use_batch_norm: bool,
        use_skip: bool = False, dropout: float = 0.0, reduction_name: str = 'max',
        *, device: str = 'cuda:0', seed: int = 0, dag_builder_class: typing.Any = PipelineDAGBuilder
    ):
        super().__init__()
        self.module_ids = module_ids
        self.n_layers = n_layers
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.activation_name = activation_name
        self._activation = get_activation(activation_name, functional=True)
        self.reduction_name = reduction_name
        self._reduction = get_reduction(reduction_name)
        self._reduction_dim = 0
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.device = device
        self.seed = seed
        self._input_seed = seed + 1
        self._output_seed = seed + 2
        self._dynamic_base_seed = seed + 3

        self._fc_input_module = self._init_fc_input_module()
        self._dynamic_modules = self._init_dynamic_modules(module_ids)
        self._dag_builder = dag_builder_class(
            self._dynamic_modules, self._reduction, self._reduction_dim, self._activation
        )
        self._dags = {}  # cache for prebuilt dags
        self._fc_output_module = self._init_fc_output_module()


        # self._daglstm = DAGLSTMModule()

    def _init_fc_input_module(self):
        layer_sizes = [self.input_layer_size] + [self.hidden_layer_size] * (self.n_layers - 1)
        return FullyConnectedModule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout,
            device=self.device, seed=self._input_seed
        )

    def _init_fc_output_module(self):
        layer_sizes = [self.hidden_layer_size] * (self.n_layers - 1) + [self.output_layer_size]
        return FullyConnectedModule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout,
            device=self.device, seed=self._output_seed
        )

    def _init_dynamic_modules(self, module_ids: typing.Sequence) -> nn.ModuleDict:
        dynamic_modules = nn.ModuleDict()
        for i, id_ in enumerate(module_ids):
            layer_sizes = [self.hidden_layer_size] + [self.hidden_layer_size] * (self.n_layers - 1)
            dynamic_modules[id_] = FullyConnectedModule(
                layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip,
                self.dropout, device=self.device, seed=self._dynamic_base_seed + i
            )
        return dynamic_modules

    def _get_dag(self, dag_id, dag_data):
        if dag_id not in self._dags:
            self._dags[dag_id] = self._dag_builder.get_dag(dag_data)
        return self._dags[dag_id]

    def forward(self, args):
        dag_id, dag_data, x = args

        # dag_structure = None # todo: what is this?
        # lstm_start_state = None # todo: what is this?
        # daglstm_outputs = self._daglstm(dag_data, dag_structure)
        h1 = self._fc_input_module(x)
        nmn_dag = self._get_dag(dag_id, dag_data)
        h2 = nmn_dag(h1)[0]

        # fc_output_inputs = torch.cat((daglstm_outputs, nmn_outputs), 1) # todo: which dimension?
        # return self._fc_output(fc_output_inputs)
        h3 = self._fc_output_module(h2)
        return torch.squeeze(h3)
