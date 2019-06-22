import torch.nn as nn
import torch

from .models import ACTIVATIONS
from .submodule import Submodule


class DAGRNN(nn.Module):
    """
    The DAGRNN can be used in both an RNN regression task or an RNN siamese task.
    It parses a pipeline DAG by saving hidden states of previously seen primitives and combining them.
    It passes the combined hidden states, which represent inputs into the next primitive, into an LSTM.
    The primitives are one hot encoded.
    """

    def __init__(
            self, activation_name: str, input_n_hidden_layers: int, input_hidden_layer_size: int, input_dropout: float,
            hidden_state_size: int, lstm_n_layers: int, lstm_dropout: float, bidirectional: bool,
            output_n_hidden_layers: int, output_hidden_layer_size: int, output_dropout: float, use_batch_norm: bool,
            use_skip: bool, rnn_input_size: int, input_layer_size: int, output_size: int, device: str, seed: int
    ):
        super(DAGRNN, self).__init__()

        self.input_layer_size = input_layer_size
        self.activation_name = activation_name
        self.use_batch_norm = use_batch_norm
        self.use_skip = use_skip
        self.output_size = output_size
        self.device = device
        self._input_seed = seed + 1
        self._output_seed = seed + 2

        self.input_n_hidden_layers = input_n_hidden_layers
        self.input_hidden_layer_size = input_hidden_layer_size
        self.input_dropout = input_dropout
        self._input_submodule = self._get_input_submodule(output_size=hidden_state_size)

        self.activation = ACTIVATIONS[activation_name]()

        if input_dropout > 0.0:
            self.input_dropout_layer = nn.Dropout(p=input_dropout)
            self.input_dropout_layer.to(device=device)

        if use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(hidden_state_size)
            self.batch_norm.to(device=self.device)

        n_directions = 2 if bidirectional else 1
        self.hidden_state_dim0_size = lstm_n_layers * n_directions
        self.lstm = nn.LSTM(
            input_size=rnn_input_size, hidden_size=hidden_state_size, num_layers=lstm_n_layers, dropout=lstm_dropout,
            bidirectional=bidirectional, batch_first=True
        )
        self.lstm.to(device=self.device)

        lstm_output_size = hidden_state_size * n_directions
        self.output_n_hidden_layers = output_n_hidden_layers
        self.output_hidden_layer_size = output_hidden_layer_size
        self.output_dropout = output_dropout
        self._output_submodule = self._get_output_submodule(input_size=lstm_output_size)

        self.NULL_INPUTS = ['inputs.0']

    def _get_input_submodule(self, output_size):
        layer_sizes = [self.input_layer_size] + [self.input_hidden_layer_size] * self.input_n_hidden_layers
        layer_sizes += [output_size]
        return Submodule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, device=self.device,
            seed=self._input_seed, dropout=self.input_dropout
        )

    def _get_output_submodule(self, input_size):
        layer_sizes = [input_size] + [self.output_hidden_layer_size] * self.output_n_hidden_layers
        layer_sizes += [self.output_size]
        return Submodule(
            layer_sizes, self.activation_name, self.use_batch_norm, self.use_skip, device=self.device,
            seed=self._output_seed, dropout=self.output_dropout
        )

    def forward(self, args):
        (pipeline_structure, pipelines, metafeatures) = args

        batch_size = pipelines.shape[0]
        seq_len = pipelines.shape[1]

        assert len(metafeatures) == batch_size
        assert len(pipeline_structure) == seq_len

        # Pass the metafeatures through the input layer
        metafeatures = self._input_submodule(metafeatures)
        metafeatures = self.activation(metafeatures)

        if self.input_dropout > 0.0:
            metafeatures = self.input_dropout_layer(metafeatures)

        if self.use_batch_norm:
            metafeatures = self.batch_norm(metafeatures)

        # Add a dimension to the metafeatures so they can be concatenated across that dimension
        metafeatures = metafeatures.unsqueeze(dim=0)

        # Initialize the hidden state and cell state using a concatenation of the metafeatures
        hidden_state = []
        for i in range(self.hidden_state_dim0_size):
            hidden_state.append(metafeatures)
        hidden_state = torch.cat(hidden_state, dim=0)
        cell_state = hidden_state
        lstm_start_state = (hidden_state, cell_state)

        # Keep track of the previous hidden and cell states as we traverse the pipeline
        prev_lstm_states = []

        lstm_output = None

        # For each batch of primitives coming from the batch of pipelines
        for i in range(seq_len):
            # Get the one hot encoded primitives from the batch at index i in the pipeline
            encoded_primitives = pipelines[:, i, :]

            # Add a dimension in the middle of the shape so it is batch size by sequence length by input size
            # Note that the sequence length (middle) is 1 because we're doing 1 (batch of) primitive(s) at a time
            # Also note that the batch size is first in the shape because we selected batch first
            encoded_primitives = encoded_primitives.unsqueeze(dim=1)

            # Get the list of indices of input primitives coming into the current primitive
            primitive_inputs = pipeline_structure[i]

            # If this is the first primitive
            if primitive_inputs == self.NULL_INPUTS:
                lstm_input_state = lstm_start_state
            else:
                # Otherwise get the mean of the hidden states of the input primitives
                lstm_input_state = self.get_lstm_input_state(prev_lstm_states=prev_lstm_states,
                                                             primitive_inputs=primitive_inputs)

            (lstm_output, lstm_output_state) = self.lstm(encoded_primitives, lstm_input_state)

            prev_lstm_states.append(lstm_output_state)

        # Since we're doing 1 primitive at a time, the sequence length in the LSTM output is 1
        # Squeeze so the fully connected input is of shape batch size by num_directions*hidden_size
        linear_input = lstm_output.squeeze(dim=1)

        linear_output = self._output_submodule(linear_input)
        return linear_output.squeeze()

    @staticmethod
    def get_lstm_input_state(prev_lstm_states, primitive_inputs):
        # Get the hidden and cell states produced by primitives connected to the current primitive as input
        input_hidden_states = []
        input_cell_states = []
        for primitive_input in primitive_inputs:
            (hidden_state, cell_state) = prev_lstm_states[primitive_input]
            input_hidden_states.append(hidden_state.unsqueeze(dim=0))
            input_cell_states.append(cell_state.unsqueeze(dim=0))

        # Average the input hidden and cell states each into a single state
        input_hidden_states = torch.cat(input_hidden_states, dim=0)
        input_cell_states = torch.cat(input_cell_states, dim=0)
        mean_hidden_state = torch.mean(input_hidden_states, dim=0)
        mean_cell_state = torch.mean(input_cell_states, dim=0)

        return (mean_hidden_state, mean_cell_state)
