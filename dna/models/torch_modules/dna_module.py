import typing

import torch
import torch.nn as nn

from .submodule import Submodule
from . import F_ACTIVATIONS

class DNAModule(nn.Module):

    def __init__(
        self, submodule_input_sizes: typing.Dict[str, int], n_layers: int, input_layer_size: int, hidden_layer_size: int,
        output_layer_size: int, activation_name: str, use_batch_norm: bool, use_skip: bool = False, dropout: float = 0.0, 
        reduction_name: str = 'max', *, device: str = 'cuda:0', seed: int = 0
    ):
        super().__init__()
        self.submodule_input_sizes = submodule_input_sizes
        self.n_layers = n_layers
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        self.activation_name = activation_name
        self._activation = F_ACTIVATIONS[activation_name]
        self.reduction_name = reduction_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.dropout = dropout
        self.device = device
        self.seed = seed
        self._input_seed = seed + 1
        self._output_seed = seed + 2
        self._dna_base_seed = seed + 3
        self._input_submodule = self._get_input_submodule()
        self._output_submodule = self._get_output_submodule()
        self._dynamic_submodules = self._get_dynamic_submodules()

    def _get_input_submodule(self):
        layer_sizes = [self.input_layer_size] + [self.hidden_layer_size] * (self.n_layers - 1)
        return Submodule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
            seed=self._input_seed
        )

    def _get_output_submodule(self):
        layer_sizes = [self.hidden_layer_size] * (self.n_layers - 1) + [self.output_layer_size]
        return Submodule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
            seed=self._output_seed
        )

    def _get_dynamic_submodules(self):
        dynamic_submodules = torch.nn.ModuleDict()
        for i, (submodule_id, submodule_input_size) in enumerate(sorted(self.submodule_input_sizes.items())):
            layer_sizes = [self.hidden_layer_size * submodule_input_size] + [self.hidden_layer_size] * (self.n_layers - 1)
            dynamic_submodules[submodule_id] = Submodule(
                layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, self.dropout, device=self.device,
                seed=self._dna_base_seed + i
            )
        return dynamic_submodules

    def forward(self, args):
        pipeline_id, pipeline, x = args
        outputs = {'inputs.0': self._input_submodule(x)}
        for i, step in enumerate(pipeline['steps']):
            inputs = self.reduction([outputs[j] for j in step['inputs']])
            submodule = self._dynamic_submodules[step['name']]
            h = self._activation(submodule(inputs))
            outputs[i] = h
        return torch.squeeze(self._output_submodule(h))

    def reduction(self, inputs: tuple):
        if len(inputs) > 1:
            # more than one tensor to aggregate
            complete_tensor = torch.stack(inputs)
            if self.reduction_name == 'concat':
                return torch.cat(tuple((inputs)), dim=1)
            elif self.reduction_name == 'max':
                def op(inputs): 
                    return torch.max(inputs, dim=0).values
            elif self.reduction_name == 'mean':
                def op(inputs): return torch.mean(inputs, dim=0)
            elif self.reduction_name == 'sum':
                def op(inputs): return torch.sum(inputs, dim=0)
            else:
                raise ValueError('Did not recognize the aggregate function: {}'.format(self.reduction_name))

            return op(complete_tensor)
        else:
            # only one tensor - no need for an aggregate
            return inputs[0]
