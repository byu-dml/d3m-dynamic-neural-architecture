import typing

import torch.nn as nn

from .dag import DAGModule


class PipelineDAGBuilder:
    """Builds a DAGModule from a pipeline JSON object."""

    def __init__(
        self, modules: nn.ModuleDict, reduction: typing.Callable, reduction_dim: int,
        activation: typing.Callable,
    ):
        self.modules = modules
        self.reduction = reduction
        self.reduction_dim = reduction_dim
        self.activation = activation

    def _get_module_from_step(self, step):
        return self.modules[step['name']]

    def _get_node_input_from_step_input(self, step_input):
        if isinstance(step_input, str) and step_input.startswith('inputs'):
            return DAGModule.NodeInput(
                DAGModule.NodeInputType.DAGInput, int(step_input.split('.')[1])
            )
        elif isinstance(step_input, int):
            return DAGModule.NodeInput(DAGModule.NodeInputType.NodeOutput, step_input)
        else:
            raise RuntimeError('invalid pipeline step input {}'.format(step_input))

    def _get_inputs_from_step(self, step):
        return  [
            self._get_node_input_from_step_input(step_input) for step_input in step['inputs']
        ]

    def _is_output(self, step_index, pipeline):
        return step_index == (len(pipeline['steps']) - 1)
        
    def get_dag(self, pipeline_json):
        dag = DAGModule()
        for step_index, step in enumerate(pipeline_json['steps']):
            module = self._get_module_from_step(step)
            inputs = self._get_inputs_from_step(step)
            is_output = self._is_output(step_index, pipeline_json)
            activation = None if is_output else self.activation
            node_index = dag.insert_node(
                module=self.modules[step['name']],
                inputs=inputs,
                is_output=is_output,
                reduction=self.reduction,
                reduction_dim=self.reduction_dim,
                activation=activation,
            )
            assert node_index == step_index
        return dag
