from enum import Enum
import typing
import warnings

import torch
import torch.nn as nn


class DAGModule(nn.Module):
    """A container analogous to torch.nn.Sequential for arbitrary directed acyclic graphs (DAGs).
    A node in the DAG may have multiple inputs, so a function must be provided to
    specify how multiple input tensors can be reduced into one.
    """

    class NodeInputType(Enum):
        DAGInput = 0
        NodeOutput = 1

    class NodeInput:
        def __init__(self, input_type: 'DAGModule.NodeInputType', input_index: int):
            self.input_type = input_type
            self.input_index = input_index

    class _NodeMetadata:
        def __init__(
            self, inputs: typing.Sequence['DAGModule.NodeInput'], is_output: bool,
            reduction: typing.Callable, reduction_dim: int, activation: typing.Callable
        ):
            self.inputs = inputs
            self.is_output = is_output
            if reduction is None:
                reduction = lambda x, d: x[0]
            self.reduction = reduction
            self.reduction_dim = reduction_dim
            self.activation = activation

    def _get_node_inputs(self, node_metadata: 'DAGModule._NodeMetadata', dag_inputs, node_outputs):
        node_inputs = []
        for input_ in node_metadata.inputs:
            if input_.input_type is DAGModule.NodeInputType.DAGInput:
                 node_inputs.append(dag_inputs[input_.input_index])
            elif input_.input_type is DAGModule.NodeInputType.NodeOutput:
                 node_inputs.append(node_outputs[input_.input_index])
            else:
                 raise RuntimeError(
                    'encountered invalid DAGModule.NodeInputType {}'.format(input_.input_type)
                 )
        return tuple(node_inputs)

    def __init__(self):
        super().__init__()

        self._nodes = nn.ModuleList()
        self._nodes_metadata = []

    def insert_node(
        self, module: nn.Module, inputs: typing.Sequence['DAGModule._NodeInput'], is_output: bool = False, reduction: typing.Callable = None, reduction_dim: int = 0, activation: typing.Callable = None
    ) -> int:
        for input_ in inputs:
            if input_.input_type not in DAGModule.NodeInputType:
                raise ValueError('unknown node input type {}'.format(input_.input_type))
            if (
                input_.input_type is DAGModule.NodeInputType.NodeOutput and
                input_.input_index >= len(self._nodes)
            ):
                raise ValueError(
                    'NodeOutput input_index must be smaller than the current number of nodes'
                )
        if is_output and activation is not None:
            warnings.warn('an activation function should not be given for an output node')

        node_index = len(self._nodes)
        self._nodes.append(module)
        self._nodes_metadata.append(self._NodeMetadata(inputs, is_output, reduction, reduction_dim, activation))
        return node_index

    def forward(self, *dag_inputs):
        node_outputs = []
        dag_outputs = []
        for node, node_metadata in zip(self._nodes, self._nodes_metadata):
            node_inputs = self._get_node_inputs(node_metadata, dag_inputs, node_outputs)
            reduced_node_input = node_metadata.reduction(
                torch.stack(node_inputs), node_metadata.reduction_dim
            )
            node_output = node(reduced_node_input)
            if node_metadata.activation is not None:
                node_output = node_metadata.activation(node_output)
            node_outputs.append(node_output)
            if node_metadata.is_output:
                dag_outputs.append(node_output)
        return tuple(dag_outputs)
