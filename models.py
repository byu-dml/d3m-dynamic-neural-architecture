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
    """
    A Dynamic Nueral Architecture model.

    Parameters:
    -----------
    input_model: nn.module,
    """

    def __init__(self, input_model, models, output_model):
        self.input_model = input_model
        self.models = models
        self.output_model = output_model

    def save(self):
        torch.save()#input_model)
        for model in models:
            torch.save()#model)
        torch.save(output_model)
