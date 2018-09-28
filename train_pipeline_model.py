import json

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset import PipelineDataset, preprocess_data
from models import PrimitiveModel, AccuracyRegressionModel


primitive_submodel_dict = {}
input_model = nn.Module
output_model = nn.Module


def instantiate_submodels(primitive_names, mf_len):
    for primitive_name in primitive_names:
        if not primitive_name in primitive_submodel_dict:
            primitive_submodel_dict[primitive_name] = PrimitiveModel(mf_len)


def build_model(primitive_names, mf_len):
    primitive_models = []
    primitive_models.append(input_model)
    for primitive_name in primitive_names:
        primitive_models.append(primitive_submodel_dict[primitive_name])
    primitive_models.append(output_model)

    return nn.Sequential(*primitive_models)
    

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    pipelines = PipelineDataset("./data/processed_data.json", device)
    mf_len = len(pipelines[0][0])

    global input_model
    input_model = PrimitiveModel(mf_len)
    global output_model
    output_model = AccuracyRegressionModel(mf_len)

    objective = torch.nn.MSELoss()
    n_epochs = 5
    # loop = tqdm(total=n_epochs, position=0)
    losses = []
    for e, epoch in enumerate(range(n_epochs)):
        # np.random.shuffle(pipelines)
        # loop = tqdm(total=len(pipelines), position=0)

        for i, (x, y, primitive_names) in enumerate(pipelines):
            instantiate_submodels(primitive_names, mf_len)
            
            pipeline_regressor = build_model(primitive_names, mf_len)
            pipeline_regressor.cuda()

            optimizer = optim.Adam(pipeline_regressor.parameters(), lr=1e-4)

            optimizer.zero_grad()

            prediction = pipeline_regressor(x)
            loss = torch.sqrt(objective(prediction, y))

            # loop.set_description('loss:{:.4f}'.format(loss.item()))
            # loop.update(1)

            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(e, n_epochs, i, len(pipelines), loss.item())
            losses.append(loss.item())

        # print progress of loss
        
    # loop.close()

    plt.plot(losses)
    plt.xlabel("training iteration")
    plt.ylabel("rmse loss")
    plt.title("Pipeline Regressor Training Loss")
    plt.show()

    plt.hist(losses, bins=20, range=(0,1))
    plt.show()


if __name__ == "__main__":
    main()
    # preprocess_data("./data/complete_pipelines_and_metafeatures.json")

