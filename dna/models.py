import json
import os
import typing

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import Dataset, GroupDataLoader, RNNDataLoader, group_json_objects
from kND import KNearestDatasets
import utils

F_ACTIVATIONS = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
ACTIVATIONS = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
ACTIVATION = 'leaky_relu'


class ModelNotFitError(Exception):
    pass


class Submodule(nn.Module):

    def __init__(self, input_layer_size, output_size, *, use_skip=False):
        super(Submodule, self).__init__()

        # todo: make constructor arguments
        activation = ACTIVATIONS[ACTIVATION]
        n_layers=1
        n_hidden_nodes=1
        batch_norms=[True]

        # The length of the batch norms list must be this size to account for the hidden layers and input layer
        assert len(batch_norms) == n_layers
        assert n_layers >= 1
        assert n_hidden_nodes >= 1

        layers = []
        if n_layers == 1:
            # Create a single without an activation function
            if batch_norms[0]:
                layers.append(nn.BatchNorm1d(input_layer_size))
            layers.append(nn.Linear(input_layer_size, output_size))
        else:
            # Create the first layer
            if batch_norms[0]:
                layers.append(nn.BatchNorm1d(input_layer_size))
            layers.append(nn.Linear(input_layer_size, n_hidden_nodes))
            layers.append(activation())

            # Create the hidden layers not including the output layer
            last_index = n_layers - 1
            for i in range(1, last_index):
                if batch_norms[i]:
                    layers.append(nn.BatchNorm1d(n_hidden_nodes))
                layers.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
                layers.append(activation())

            # Create the output layer without an activation function
            if batch_norms[last_index]:
                layers.append(nn.BatchNorm1d(n_hidden_nodes))
            layers.append(nn.Linear(n_hidden_nodes, output_size))

        self.net = nn.Sequential(*layers)

        if use_skip:
            if input_layer_size == output_size:
                self.skip = nn.Sequential()
            else:
                self.skip = nn.Linear(input_layer_size, output_size)
        else:
            self.skip = None

    def forward(self, x):
        if self.skip:
            return self.net(x) + self.skip(x)
        else:
            return self.net(x)


class DNAModule(nn.Module):

    def __init__(
        self, input_model=None, submodules=None, output_model=None, *, seed=0
    ):
        # todo use seed
        super(DNAModule, self).__init__()
        self.input_model = input_model
        self.submodules = submodules
        self.output_model = output_model
        self.h1 = None
        self.f_activation = F_ACTIVATIONS[ACTIVATION]

    def forward(self, args):
        # pipeline_id, pipeline, x, datasets = args
        pipeline_id, pipeline, x = args
        self.h1 = self.f_activation(self.input_model(x))
        h2 = self.f_activation(self.recursive_get_output(pipeline['steps'], len(pipeline['steps']) - 1))
        return torch.squeeze(self.output_model(h2))

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        path = os.path.join(save_dir, "input_model.pt")
        self._save(self.input_model, path)

        for name, model in self.submodules.items():
            path = os.path.join(save_dir, f"{name}_model.pt")
            self._save(model, path)

        path = os.path.join(save_dir, "output_model.pt")
        self._save(self.output_model, path)

    def _save(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def load(self, save_dir):
        if not os.path.isdir(save_dir):
            raise ValueError(f"save_dir {save_dir} does not exist")

        path = os.path.join(save_dir, "input_model.pt")
        self._load(self.input_model, path)

        for name, model in self.submodules.items():
            path = os.path.join(save_dir, f"{name}_model.pt")
            self._load(model, path)

        path = os.path.join(save_dir, "output_model.pt")
        self._load(self.output_model, path)

    def _load(self, model, path):
        model.load_state_dict(torch.load(path))

    def recursive_get_output(self, pipeline, current_index):
        """
        The recursive call to find the input
        :param pipeline: the pipeline list containing the submodules
        :param current_index: the index of the current submodule
        :return:
        """
        current_submodule = self.submodules[pipeline[current_index]["name"]]
        if "inputs.0" in pipeline[current_index]["inputs"]:
            return self.f_activation(current_submodule(self.h1))

        outputs = []
        for input in pipeline[current_index]["inputs"]:
            curr_output = self.recursive_get_output(pipeline, input)
            outputs.append(curr_output)

        if len(outputs) > 1:
            new_output = self.f_activation(current_submodule(torch.cat(tuple(outputs), dim=1)))
        else:
            new_output = self.f_activation(current_submodule(curr_output))

        return new_output


class ModelBase:

    def __init__(self, *, seed):
        self.seed = seed
        self.fitted = False

    def fit(self, data, *, verbose=False):
        raise NotImplementedError()


class RegressionModelBase(ModelBase):

    def predict_regression(self, data, *, verbose=False):
        raise NotImplementedError()


class RankModelBase(ModelBase):

    def predict_rank(self, data, k=None, *, verbose=False):
        raise NotImplementedError()


class PyTorchModelBase:

    def __init__(self, task_type, *, seed, device):
        if task_type == "CLASSIFICATION":
            self._y_dtype = torch.int64
        elif task_type == "REGRESSION":
            self._y_dtype = torch.float32
        self.seed = seed
        self.device = device
        self._model = None

    def fit(
        self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
        verbose=False
    ):
        self._model = self._get_model(train_data)
        self._optimizer = self._get_optimizer(learning_rate)

        train_data_loader = self._get_data_loader(train_data, batch_size, drop_last)
        validation_data_loader = None
        if validation_data is not None:
            validation_data_loader = self._get_data_loader(validation_data, batch_size, False)

        for e in range(n_epochs):
            if verbose:
                print('epoch {}'.format(e))

            self._train_epoch(
                train_data_loader, self._model, self._loss_function, self._optimizer, verbose=verbose
            )
            self._model.save(os.path.join(output_dir, 'weights'))

            train_predictions, train_targets = self._predict_epoch(train_data_loader, self._model, verbose=verbose)
            train_loss_score = self._loss_function(train_predictions, train_targets)
            self._save_outputs(output_dir, 'train', e, train_predictions, train_targets, train_loss_score)
            if verbose:
                print('train loss: {}'.format(train_loss_score))

            if validation_data_loader is not None:
                validation_predictions, validation_targets = self._predict_epoch(validation_data_loader, self._model, verbose=verbose)
                validation_loss_score = self._loss_function(validation_predictions, validation_targets)
                self._save_outputs(output_dir, 'validation', e, validation_predictions, validation_targets, validation_loss_score)
                if verbose:
                    print('validation loss: {}'.format(validation_loss_score))

        self.fitted = True

    def _get_model(self, train_data):
        raise NotImplementedError()

    def _get_optimizer(self, learning_rate):
        raise NotImplementedError()

    def _get_data_loader(self, data, batch_size, drop_last):
        raise NotImplementedError()

    def _train_epoch(
        self, data_loader, model: nn.Module, loss_function, optimizer, *, verbose=True
    ):
        model.train()

        if verbose:
            progress = tqdm(total=len(data_loader), position=0)

        for x_batch, y_batch in data_loader:
            y_hat_batch = model(x_batch)
            loss = loss_function(y_hat_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

    def _predict_epoch(
        self, data_loader, model: nn.Module, *, verbose=True
    ):
        model.eval()
        predictions = []
        targets = []

        if verbose:
            progress = tqdm(total=len(data_loader), position=0)

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                y_hat_batch = model(x_batch)

                if y_batch.shape[0] == 1:
                    predictions.append(y_hat_batch.item())
                    targets.append(y_batch.item())
                else:
                    predictions.extend(y_hat_batch.tolist())
                    targets.extend(y_batch.tolist())

                if verbose:
                    progress.update(1)

        if verbose:
            progress.close()

        return torch.tensor(predictions, dtype=self._y_dtype), torch.tensor(targets, dtype=self._y_dtype)

    @staticmethod
    def _save_outputs(output_dir, phase, epoch, predictions, targets, loss_score):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_filename = phase + '_scores.csv'
        save_path = os.path.join(output_dir, save_filename)
        with open(save_path, 'a') as f:
            f.write(str(float(loss_score)) + '\n')

        output_dir = os.path.join(output_dir, 'outputs')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_filename = str(epoch) + '_' + phase + '.json'
        save_path = os.path.join(output_dir, save_filename)
        outputs = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
        }
        with open(save_path, 'w') as f:
            json.dump(outputs, f, separators=(',',':'))


class DNARegressionModel(PyTorchModelBase, RegressionModelBase, RankModelBase):

    def __init__(self, latent_size=50, *, seed, device='cuda:0'):
        self._task_type = 'REGRESSION'
        PyTorchModelBase.__init__(self, task_type=self._task_type, seed=seed, device=device)
        RegressionModelBase.__init__(self, seed=seed)
        RankModelBase.__init__(self, seed=seed)

        self.latent_size = latent_size

        objective = torch.nn.MSELoss(reduction="mean")
        self._loss_function = lambda y_hat, y: torch.sqrt(objective(y_hat, y))

    def _get_model(self, train_data):
        self.shape = (len(train_data[0]['metafeatures']), self.latent_size, 1)
        torch_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed + 1)

        input_submodule = Submodule(self.shape[0], self.shape[1])

        submodules = torch.nn.ModuleDict()
        for instance in train_data:
            for step in instance['pipeline']['steps']:
                if not step['name'] in submodules:
                    n_inputs = len(step['inputs'])
                    submodules[step['name']] = Submodule(
                        n_inputs * self.shape[1], self.shape[1]
                    )
                    submodules[step['name']].cuda()  # todo dynamically set device

        output_submodule = Submodule(self.shape[1], self.shape[2], use_skip=False)

        model = DNAModule(input_submodule, submodules, output_submodule)
        model.cuda()  # todo dynamically set device

        torch.random.set_rng_state(torch_state)

        return model

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last):
        return GroupDataLoader(
            data = data,
            group_key = 'pipeline.id',
            dataset_class = Dataset,
            dataset_params = {
                'features_key': 'metafeatures',
                'target_key': 'test_f1_macro',
                'task_type': self._task_type,
                'device': self.device
            },
            batch_size = batch_size,
            drop_last = drop_last,
            shuffle = True,
            seed = self.seed + 2
        )

    def predict_regression(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, batch_size, False)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)

        return predictions

    def predict_rank(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, batch_size, False)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)
        ranks = utils.rank(np.array(predictions))
        return {
            'pipeline_id': [instance['pipeline']['id'] for instance in data],
            'rank': ranks,
        }


class DAGRNN(nn.Module):
    """
    The DAG RNN, like the DNA Module, can be used in both an RNN regression task or an RNN siamese task.
    It parses a pipeline DAG by saving hidden states of previously seen primitives and combining them.
    It passes the combined hidden states, which represent inputs into the next primitive, into an LSTM.
    The primitives are one hot encoded.
    """
    def __init__(self, rnn_input_size, hidden_state_size, output_layer_size, n_layers, dropout, bidirectional):
        super(DAGRNN, self).__init__()

        n_directions = 2 if bidirectional else 1
        self.hidden_state_dim0_size = n_layers * n_directions

        self.lstm = nn.LSTM(input_size=rnn_input_size, hidden_size=hidden_state_size, num_layers=n_layers,
                            dropout=dropout, bidirectional=bidirectional, batch_first=True)

        lstm_output_size = hidden_state_size * n_directions
        self.linear_out = Submodule(lstm_output_size, output_layer_size)

        self.NULL_INPUTS = ['inputs.0']

    def forward(self, args):
        (pipeline_structure, pipelines, metafeatures) = args

        batch_size = pipelines.shape[0]
        seq_len = pipelines.shape[1]

        assert len(metafeatures) == batch_size
        assert len(pipeline_structure) == seq_len

        # TODO: Consider passing the metafeatures through at least one dense layer so hidden size is not restricted
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

        linear_output = self.linear_out(linear_input)
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

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        path = os.path.join(save_dir, "dag_rnn.pt")
        torch.save(self.state_dict(), path)

    def load(self, save_dir):
        if not os.path.isdir(save_dir):
            raise ValueError(f"save_dir {save_dir} does not exist")

        path = os.path.join(save_dir, "dag_rnn.pt")
        self.load_state_dict(torch.load(path))


class RNNRegressionModel(PyTorchModelBase, RegressionModelBase, RankModelBase):
    def __init__(self, latent_size=50, *, seed, device='cuda:0'):
        self._task_type = 'REGRESSION'
        PyTorchModelBase.__init__(self, task_type=self._task_type, seed=seed, device=device)
        RegressionModelBase.__init__(self, seed=seed)
        RankModelBase.__init__(self, seed=seed)

        # TODO: Consider making the latent size the hidden state size and have a dense layer process the metafeatures
        # TODO: The input size of that dense layer should be num mfs and output size should be hidden state size
        self.latent_size = latent_size

        objective = torch.nn.MSELoss(reduction="mean")
        self._loss_function = lambda y_hat, y: torch.sqrt(objective(y_hat, y))

        self.pipeline_structures = None
        self.num_primitives = None
        self.target_key = 'test_f1_macro'
        self.batch_group_key = 'pipeline_structure'
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.prim_name_key = 'name'
        self.prim_inputs_key = 'inputs'
        self.features_key = 'metafeatures'

    def fit(self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
        verbose=False):
        # Get all the pipeline structure for each pipeline structure group before encoding the pipelines
        self.pipeline_structures = {}
        grouped_by_structure = group_json_objects(train_data, self.batch_group_key)
        for (group, group_indices) in grouped_by_structure.items():
            index = group_indices[0]
            item = train_data[index]
            pipeline = item[self.pipeline_key][self.steps_key]
            group_structure = [primitive[self.prim_inputs_key] for primitive in pipeline]
            self.pipeline_structures[group] = group_structure

        # Get the mapping of primitives to their one hot encoding
        primitive_name_to_enc = self._get_primitive_name_to_enc(train_data=train_data)

        # Encode all the pipelines in the training and validation set
        self.encode_pipelines(data=train_data, primitive_name_to_enc=primitive_name_to_enc)
        if validation_data is not None:
            self.encode_pipelines(data=validation_data, primitive_name_to_enc=primitive_name_to_enc)

        print('Ready to create model and start training')

        PyTorchModelBase.fit(self, train_data, n_epochs, learning_rate, batch_size, drop_last,
                             validation_data=validation_data, output_dir=output_dir, verbose=verbose)

    def _get_primitive_name_to_enc(self, train_data):
        primitive_names = set()

        # Get a set of all the primitives in the train set
        for instance in train_data:
            primitives = instance[self.pipeline_key][self.steps_key]
            for primitive in primitives:
                primitive_name = primitive[self.prim_name_key]
                primitive_names.add(primitive_name)

        # Get one hot encodings of all the primitives
        self.num_primitives = len(primitive_names)
        encoding = np.identity(n=self.num_primitives)

        # Create a mapping of primitive names to one hot encodings
        primitive_name_to_enc = {}
        for (primitive_name, primitive_encoding) in zip(primitive_names, encoding):
            primitive_name_to_enc[primitive_name] = primitive_encoding

        return primitive_name_to_enc

    def encode_pipelines(self, data, primitive_name_to_enc):
        for instance in data:
            pipeline = instance[self.pipeline_key][self.steps_key]
            encoded_pipeline = self.encode_pipeline(pipeline=pipeline, primitive_to_enc=primitive_name_to_enc)
            instance[self.pipeline_key][self.steps_key] = encoded_pipeline

    def encode_pipeline(self, pipeline, primitive_to_enc):
        # Create a tensor of encoded primitives
        encoding = []
        for primitive in pipeline:
            primitive_name = primitive[self.prim_name_key]
            encoded_primitive = primitive_to_enc[primitive_name]
            encoding.append(encoded_primitive)

        encoding = torch.tensor(encoding, dtype=torch.float32)

        if "cuda" in self.device:
            encoding = encoding.cuda()

        return encoding

    def _get_model(self, train_data):
        metafeatures_length = len(train_data[0][self.features_key])
        model = DAGRNN(rnn_input_size=self.num_primitives, hidden_state_size=metafeatures_length, output_layer_size=1,
                       n_layers=2, dropout=0.1, bidirectional=True)
        model.cuda()
        return model

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last):
        dataset_params = {
            "features_key": self.features_key,
            "target_key": self.target_key,
            "task_type": self._task_type,
            "device": self.device
        }

        return RNNDataLoader(data=data, group_key=self.batch_group_key, dataset_params=dataset_params,
                             batch_size=batch_size, drop_last=drop_last, shuffle=True, seed=self.seed,
                             pipeline_structures=self.pipeline_structures)

    def predict_regression(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, batch_size, False)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)

        return predictions

    def predict_rank(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, batch_size, False)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)
        ranks = utils.rank(np.array(predictions))
        return {
            'pipeline_id': [instance['pipeline']['id'] for instance in data],
            'rank': ranks,
        }


class DNASiameseModule(nn.Module):

    def __init__(self, input_model, submodules, output_model):
        super(DNASiameseModule, self).__init__()
        self.input_model = input_model
        self.submodules = submodules
        self.output_model = output_model
        self.h1 = None
        self.f_activation = F_ACTIVATIONS[ACTIVATION]

    def forward(self, args):
        pipeline_ids, (left_pipeline, right_pipeline), x = args
        self.h1 = self.input_model(x)
        left_h2 = self.recursive_get_output(left_pipeline, len(left_pipeline) - 1)
        right_h2 = self.recursive_get_output(right_pipeline, len(right_pipeline) - 1)
        h2 = torch.cat((left_h2, right_h2), 1)
        return self.output_model(h2)

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        path = os.path.join(save_dir, "input_model.pt")
        self._save(self.input_model, path)

        for name, model in self.submodules.items():
            path = os.path.join(save_dir, f"{name}_model.pt")
            self._save(model, path)

        path = os.path.join(save_dir, "output_model.pt")
        self._save(self.output_model, path)

    def _save(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def load(self, save_dir):
        if not os.path.isdir(save_dir):
            raise ValueError(f"save_dir {save_dir} does not exist")

        path = os.path.join(save_dir, "input_model.pt")
        self._load(self.input_model, path)

        for name, model in self.submodules.items():
            path = os.path.join(save_dir, f"{name}_model.pt")
            self._load(model, path)

        path = os.path.join(save_dir, "output_model.pt")
        self._load(self.output_model, path)

    def _load(self, model, path):
        model.load_state_dict(torch.load(path))

    def recursive_get_output(self, pipeline, current_index):
        """
        The recursive call to find the input
        :param pipeline: the pipeline list containing the submodules
        :param current_index: the index of the current submodule
        :return:
        """
        try:
            current_submodule = self.submodules[pipeline[current_index]['name']]
            if "inputs.0" in pipeline[current_index]['inputs']:
                return self.f_activation(current_submodule(self.h1))

            outputs = []
            for input in pipeline[current_index]["inputs"]:
                curr_output = self.recursive_get_output(pipeline, input)
                outputs.append(curr_output)

            if len(outputs) > 1:
                new_output = self.f_activation(current_submodule(torch.cat(tuple(outputs), dim=1)))
            else:
                new_output = self.f_activation(current_submodule(curr_output))

            return new_output
        except Exception as e:
            print("There was an error in the foward pass.  It was ", e)
            print(pipeline[current_index])
            quit(1)


class MeanBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        RegressionModelBase.__init__(self, seed=seed)
        self.mean = None

    def fit(self, data, *, validation_data=None, output_dir=None, verbose=False):
        total = 0
        for instance in data:
            total += instance['test_f1_macro']
        self.mean = total / len(data)
        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.mean is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.mean] * len(data)


class MedianBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        RegressionModelBase.__init__(self, seed=seed)
        self.median = None

    def fit(self, data, *, validation_data=None, output_dir=None, verbose=False):
        self.median = np.median([instance['test_f1_macro'] for instance in data])
        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.median is None:
            raise ModelNotFitError('MeanBaseline not fit')
        return [self.median] * len(data)


class PerPrimitiveBaseline(RegressionModelBase):

    def __init__(self, seed=0):
        RegressionModelBase.__init__(self, seed=seed)
        self.primitive_scores = None

    def fit(self, data, *, validation_data=None, output_dir=None, verbose=False):
        # for each primitive, get the scores of all the pipelines that use the primitive
        primitive_score_totals = {}
        for instance in data:
            for primitive in instance['pipeline']['steps']:
                if primitive['name'] not in primitive_score_totals:
                    primitive_score_totals[primitive['name']] = {
                        'total': 0,
                        'count': 0,
                    }
                primitive_score_totals[primitive['name']]['total'] += instance['test_f1_macro']
                primitive_score_totals[primitive['name']]['count'] += 1

        # compute the average pipeline score per primitive
        self.primitive_scores = {}
        for primitive_name in primitive_score_totals:
            total = primitive_score_totals[primitive_name]['total']
            count = primitive_score_totals[primitive_name]['count']
            self.primitive_scores[primitive_name] = total / count

        self.fitted = True

    def predict_regression(self, data, *, verbose=False):
        if self.primitive_scores is None:
            raise ModelNotFitError('PerPrimitiveBaseline not fit')

        predictions = []
        for instance in data:
            prediction = 0
            for primitive in instance['pipeline']['steps']:
                prediction += self.primitive_scores[primitive['name']]
            prediction /= len(instance['pipeline'])
            predictions.append(prediction)

        return predictions


class AutoSklearnMetalearner(RankModelBase):

    def __init__(self, seed=0):
        RankModelBase.__init__(self, seed=seed)

    def get_k_best_pipelines(self, data, dataset_metafeatures, all_other_metafeatures, runs, current_dataset_name):
        # all_other_metafeatures = all_other_metafeatures.iloc[:, mf_mask]
        all_other_metafeatures = all_other_metafeatures.replace([np.inf, -np.inf], np.nan)
        # this should aready be done by the time it gets here
        all_other_metafeatures = all_other_metafeatures.transpose()
        # get the metafeatures out of their list
        all_other_metafeatures = pd.DataFrame(all_other_metafeatures.iloc[1].tolist(), index=all_other_metafeatures.iloc[0])
        all_other_metafeatures = all_other_metafeatures.fillna(all_other_metafeatures.mean(skipna=True))
        all_other_metafeatures = all_other_metafeatures.reset_index().drop_duplicates()
        all_other_metafeatures = all_other_metafeatures.set_index('dataset_id')
        # get the ids for pipelines that we have real values for
        current_validation_ids = set(pipeline['id'] for pipeline in data.pipeline)

        kND = KNearestDatasets(metric='l1', random_state=3)
        kND.fit(all_other_metafeatures, self.run_lookup, current_validation_ids, self.maximize_metric)
        # best suggestions is a list of 3-tuples that contain the pipeline index,the distance value, and the pipeline_id
        best_suggestions = kND.kBestSuggestions(pd.Series(dataset_metafeatures), k=all_other_metafeatures.shape[0])
        k_best_pipelines = [suggestion[2] for suggestion in best_suggestions]
        return k_best_pipelines

    def get_k_best_pipelines_per_dataset(self, data):
        # they all should have the same dataset and metafeatures so take it from the first row
        dataset_metafeatures = data["metafeatures"].iloc[0]
        dataset_name = data["dataset_id"].iloc[0]
        all_other_metafeatures = self.metafeatures
        pipelines = self.get_k_best_pipelines(data, dataset_metafeatures, all_other_metafeatures, self.runs, dataset_name)
        return pipelines


    def predict_rank(self, data, *, verbose=False):
        """
        A wrapper for all the other functions so that this is organized
        :data: a dictionary containing pipelines, ids, and real f1 scores. MUST CONTAIN PIPELINE IDS
        from each dataset being passed in.  This is used for the rankings
        :return:
        """
        data = pd.DataFrame(data)
        k_best_pipelines_per_dataset = self.get_k_best_pipelines_per_dataset(data)
        return {
            'pipeline_id': k_best_pipelines_per_dataset,
            'rank': list(range(len(k_best_pipelines_per_dataset))),
        }

    def fit(self, training_dataset=None, metric='test_accuracy', maximize_metric=True, *, validation_data=None, output_dir=None, verbose=False):
        """
        A basic KNN fit.  Loads in and processes the training data from a fixed split
        :param training_dataset: the dataset to be processed.  If none given it will be pulled from the hardcoded file
        :param metric: what kind of metric we're using in our metalearning
        :param maximize_metric: whether to maximize or minimize that metric.  Defaults to Maximize
        """
        # if metadata_path is None:
        self.runs = None
        self.test_runs = None
        self.metafeatures = None
        self.datasets = []
        self.testset = []
        self.pipeline_descriptions = {}
        self.metric = metric
        self.maximize_metric = maximize_metric
        self.opt = np.nanmax
        if training_dataset is None:
            # these are in this order so the metadata holds the train and self.datasets and self.testsets get filled
            with open(os.path.join(os.getcwd(), "dna/data", "test_data.json"), 'r') as f:
                self.metadata = json.load(f)
            self.process_metadata(data_type="test")
            with open(os.path.join(os.getcwd(), "dna/data", "train_data.json"), 'r') as f:
                self.metadata = json.load(f)
            self.process_metadata(data_type="train")
        else:
            self.metadata = training_dataset
            self.metafeatures = pd.DataFrame(self.metadata)[['dataset_id', 'metafeatures']]
            self.runs = pd.DataFrame(self.metadata)[['dataset_id', 'pipeline', 'test_f1_macro']]
            self.run_lookup = self.process_runs()

    def process_runs(self):
        """
        This function is used to transform the dataframe into a workable object fot the KNN, with rows of pipeline_ids
        and columns of datasets, with the inside being filled with the scores
        :return:
        """
        new_runs = {}
        for index, row in self.runs.iterrows():
            dataset_name = row["dataset_id"]
            if dataset_name not in new_runs:
                new_runs[dataset_name] = {}
            else:
                new_runs[dataset_name][row["pipeline"]['id']] = row['test_f1_macro']
        final_new = pd.DataFrame(new_runs)
        return final_new


def get_model(model_name: str, model_config: typing.Dict, seed: int):
    model_class = {
        'dna_regression': DNARegressionModel,
        'mean_regression': MeanBaseline,
        'median_regression': MedianBaseline,
        'per_primitive_regression': PerPrimitiveBaseline,
        'autosklearn': AutoSklearnMetalearner,
        'rnn_regression': RNNRegressionModel
    }[model_name.lower()]
    init_model_config = model_config.get('__init__', {})
    return model_class(seed=seed)
