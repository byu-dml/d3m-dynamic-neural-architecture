import torch.nn as nn


class PrimitiveModel(nn.Module):
    def __init__(self, input_size):
        super(PrimitiveModel, self).__init__()
        output_size = 4

        # todo with brandon: define generic submodel
        self.net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.Linear(output_size, output_size)
        )

    def forward(self, *x):
        # todo: define forward with brandon
        return self.net(x)


class PipelineRegressor(nn.Module):
    def __init__(self, ):
        super(PipelineRegressor, self).__init__()

    def forward(self, *x):
        pass