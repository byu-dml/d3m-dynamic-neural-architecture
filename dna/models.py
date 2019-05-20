import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class DNAModel(nn.Module):

    def __init__(self, input_model, submodels, output_model):
        super(DNAModel, self).__init__()
        self.input_model = input_model
        self.submodels = submodels
        self.output_model = output_model
        self.h1 = None
        self.f_activation = F_ACTIVATIONS[ACTIVATION]

    def forward(self, args):
        pipeline_id, pipeline, x = args
        x = x[0]
        self.h1 = self.f_activation(self.input_model(x))
        h2 = self.f_activation(self.recursive_get_output(pipeline, len(pipeline) - 1))
        return torch.squeeze(self.output_model(h2))

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        path = os.path.join(save_dir, "input_model.pt")
        self._save(self.input_model, path)

        for name, model in self.submodels.items():
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

        for name, model in self.submodels.items():
            path = os.path.join(save_dir, f"{name}_model.pt")
            self._load(model, path)

        path = os.path.join(save_dir, "output_model.pt")
        self._load(self.output_model, path)

    def _load(self, model, path):
        model.load_state_dict(torch.load(path))

    def recursive_get_output(self, pipeline, current_index):
        """
        The recursive call to find the input
        :param pipeline: the pipeline list containing the submodels
        :param current_index: the index of the current submodel
        :return:
        """
        try:
            current_submodel = self.submodels[pipeline[current_index]["name"]]
            if "inputs.0" in pipeline[current_index]["inputs"]:
                return self.f_activation(current_submodel(self.h1))

            outputs = []
            for input in pipeline[current_index]["inputs"]:
                curr_output = self.recursive_get_output(pipeline, input)
                outputs.append(curr_output)

            if len(outputs) > 1:
                new_output = self.f_activation(current_submodel(torch.cat(tuple(outputs), dim=1)))
            else:
                new_output = self.f_activation(current_submodel(curr_output))

            return new_output
        except Exception as e:
            print("There was an error in the foward pass.  It was ", e)
            print(pipeline[current_index])
            quit(1)


class SiameseModel(nn.Module):

    def __init__(self, input_model, submodels, output_model):
        super(SiameseModel, self).__init__()
        self.input_model = input_model
        self.submodels = submodels
        self.output_model = output_model
        self.h1 = None
        self.f_activation = F_ACTIVATIONS[ACTIVATION]

    def forward(self, args):
        pipeline_id, (left_pipeline, right_pipeline), x = args
        x = x[0]
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

        for name, model in self.submodels.items():
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

        for name, model in self.submodels.items():
            path = os.path.join(save_dir, f"{name}_model.pt")
            self._load(model, path)

        path = os.path.join(save_dir, "output_model.pt")
        self._load(self.output_model, path)

    def _load(self, model, path):
        model.load_state_dict(torch.load(path))

    def recursive_get_output(self, pipeline, current_index):
        """
        The recursive call to find the input
        :param pipeline: the pipeline list containing the submodels
        :param current_index: the index of the current submodel
        :return:
        """
        try:
            current_submodel = self.submodels[pipeline[current_index]["name"]]
            if "inputs.0" in pipeline[current_index]["inputs"]:
                return self.f_activation(current_submodel(self.h1))

            outputs = []
            for input in pipeline[current_index]["inputs"]:
                curr_output = self.recursive_get_output(pipeline, input)
                outputs.append(curr_output)

            if len(outputs) > 1:
                new_output = self.f_activation(current_submodel(torch.cat(tuple(outputs), dim=1)))
            else:
                new_output = self.f_activation(current_submodel(curr_output))

            return new_output
        except Exception as e:
            print("There was an error in the foward pass.  It was ", e)
            print(pipeline[current_index])
            quit(1)
