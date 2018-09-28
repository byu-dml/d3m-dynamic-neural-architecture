import torch
import torch.nn as nn


class PrimitiveModel(nn.Module):

    def __init__(self, visible_layer_size):
        super(PrimitiveModel, self).__init__()

        self.net = nn.Sequential(
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


class AccuracyRegressionModel(nn.Module):

    def __init__(self, input_size):
        super(AccuracyRegressionModel, self).__init__()

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
                input_size, 1
            )
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x) # torch.clamp(self.net(x), 0, 1)
