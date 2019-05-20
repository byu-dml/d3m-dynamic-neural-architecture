import json
import os
import typing

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import Dataset, GroupDataLoader
import utils


F_ACTIVATIONS = {'relu': F.relu, 'leaky_relu': F.leaky_relu, 'sigmoid': F.sigmoid, 'tanh': F.tanh}
ACTIVATIONS = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
ACTIVATION = 'relu'


class Submodule(nn.Module):

    def __init__(self, input_layer_size, output_size, *, use_skip=False):
        super(Submodule, self).__init__()

        n_layers = 1
        n_hidden_nodes = 1
        batch_norms = [True]
        activation = ACTIVATIONS[ACTIVATION]

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
        pipeline_id, pipeline, x = args
        x = x
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

    def _get_optimzer(self, learning_rate):
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
            json.dump(outputs, f)


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

    def predict_rank(self, data, k=None, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        if k is None:
            k = len(data)

        data_loader = self._get_data_loader(data, batch_size, False)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)

        return utils.rank(np.array(predictions))[:k]


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

def get_model(model_name: str, model_config: typing.Dict, seed: int):
    model_class = {
        'dna_regression': DNARegressionModel,
    }[model_name.lower()]
    init_model_config = model_config.get('__init__', {})
    return model_class(seed=seed)
