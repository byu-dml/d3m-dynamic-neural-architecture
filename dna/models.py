import os
import json

import torch
import torch.nn as nn


class PrimitiveModel(nn.Module):

    def __init__(self, name, visible_layer_size):
        super(PrimitiveModel, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(visible_layer_size),
            nn.Linear(
                visible_layer_size, visible_layer_size
            ),
            nn.ReLU(),
            nn.Linear(
                visible_layer_size, visible_layer_size
            ),
            nn.ReLU(),
            nn.Linear(
                visible_layer_size, visible_layer_size
            ),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class RegressionModel(nn.Module):

    def __init__(self, input_size):
        super(RegressionModel, self).__init__()

        self.net = nn.Sequential(
            # nn.BatchNorm1d(input_size),
            nn.Linear(
                input_size, input_size
            ),
            nn.ReLU(),
            nn.Linear(
                input_size, input_size
            ),
            nn.ReLU(),
            nn.Linear(
                input_size, 1
            ),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x) # torch.clamp(self.net(x), 0, 1) #


class ClassificationModel(nn.Module):

    def __init__(self, input_size, output_size):
        super(ClassificationModel, self).__init__()

        self.net = nn.Sequential(
            # nn.BatchNorm1d(input_size),
            nn.Linear(
                input_size, input_size
            ),
            nn.ReLU(),
            nn.Linear(
                input_size, input_size
            ),
            nn.ReLU(),
            nn.Linear(
                input_size, output_size
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x) # torch.clamp(self.net(x), 0, 1)


class DNAModel(nn.Module):

    def __init__(self, input_model, submodels, output_model):
        super(DNAModel, self).__init__()
        self.input_model = input_model
        self.submodels = submodels
        self.output_model = output_model
        self.h1 = None

    def forward(self, args):
        pipeline, x = args
        self.h1 = self.input_model(x)
        # dynamically constructs dag
        h2 = self.recursive_get_output(pipeline, len(pipeline) - 1)
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
        current_submodel = self.submodels[pipeline[current_index]["name"]]
        if "inputs.0" in pipeline[current_index]["inputs"]:
            return current_submodel(self.h1)

        outputs = []
        for input in pipeline[current_index]["inputs"]:
            curr_output = self.recursive_get_output(pipeline, input)
            outputs.append(curr_output)
        new_output = current_submodel.cat(tuple(outputs), dim=0)
        return new_output


class SiameseModel(nn.Module):

    def __init__(self, input_model, submodels, output_model):
        super(SiameseModel, self).__init__()
        self.input_model = input_model
        self.submodels = submodels
        self.output_model = output_model

    def forward(self, args):
        (left_pipeline, right_pipeline), x = args
        h1 = self.input_model(x)
        left_model = nn.Sequential(
            *[self.submodels[name] for name in left_pipeline.split("___")]
        )
        right_model = nn.Sequential(
            *[self.submodels[name] for name in right_pipeline.split("___")]
        )
        left_h2 = left_model(h1)
        right_h2 = right_model(h1)
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