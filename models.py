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


class SiameseModel(nn.Module):

    def __init__(self, input_model, submodels, output_model):
        super(SiameseModel, self).__init__()
        self.input_model = input_model
        self.submodels = submodels
        self.output_model = output_model

    def forward(self, x, left_pipeline, right_pipeline):
        h1 = self.input_model(x)
        left_model = nn.Sequential(
            *[self.submodels[name] for name in left_pipeline.split("___")]
        )
        left_model.cuda()
        right_model = nn.Sequential(
            *[self.submodels[name] for name in right_pipeline.split("___")]
        )
        right_model.cuda()
        left_h2 = left_model(h1)
        right_h2 = right_model(h1)
        h2 = torch.cat((left_h2, right_h2), 1)
        return self.output_model(h2)
