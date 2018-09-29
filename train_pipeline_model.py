import json

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset import PipelineDataset, GroupDataLoader
from models import PrimitiveModel, AccuracyRegressionModel


primitive_submodel_dict = {}
input_model = nn.Module
output_model = nn.Module


def instantiate_submodels(pipeline, mf_len):
    primitive_names = pipeline.split("___")
    for primitive_name in primitive_names:
        if not primitive_name in primitive_submodel_dict:
            primitive_submodel_dict[primitive_name] = PrimitiveModel(mf_len)


def build_model(pipeline, mf_len):
    primitive_names = pipeline.split("___")
    primitive_models = []
    primitive_models.append(input_model)
    for primitive_name in primitive_names:
        primitive_models.append(primitive_submodel_dict[primitive_name])
    primitive_models.append(output_model)

    return nn.Sequential(*primitive_models)


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    dataloader = GroupDataLoader(
        path = "./data/processed_data.json",
        group_label = "pipeline",
        dataset_class = PipelineDataset,
        dataset_params = {
            "feature_label": "metafeatures",
            "target_label": "test_accuracy",
            "device": "cuda:0"
        },
        batch_size = 48
    )
    input_size = next(iter(dataloader))[1][0].shape[1]


    global input_model
    input_model = PrimitiveModel(input_size)
    global output_model
    output_model = AccuracyRegressionModel(input_size)

    objective = torch.nn.MSELoss(reduction="elementwise_mean")
    n_epochs = 5000
    # loop = tqdm(total=n_epochs, position=0)
    losses = []
    all_epoch_losses = []
    for e, epoch in enumerate(range(n_epochs)):
        # np.random.shuffle(pipelines)
        # loop = tqdm(total=len(pipelines), position=0)
        epoch_losses = []
        for i, (pipeline, (x_batch, y_batch)) in enumerate(dataloader):
            instantiate_submodels(pipeline, input_size)

            pipeline_regressor = build_model(pipeline, input_size)
            pipeline_regressor.cuda()

            optimizer = optim.Adam(pipeline_regressor.parameters(), lr=1e-4)

            optimizer.zero_grad()

            y_hat_batch = torch.squeeze(pipeline_regressor(x_batch))
            loss = torch.sqrt(objective(y_hat_batch, y_batch))

            # loop.set_description('loss:{:.4f}'.format(loss.item()))
            # loop.update(1)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            losses.append(loss.item())

            print(e, n_epochs, i, len(dataloader), loss.item())

        all_epoch_losses.append(epoch_losses)
        # print progress of loss

        # loop.close()

        plt.plot(losses)
        plt.xlabel("training iteration")
        plt.ylabel("rmse loss")
        plt.title("Pipeline Regressor Training Loss")
        plt.savefig("all_losses.png")
        plt.clf()

        avg_epoch_losses = list(map(lambda x: np.mean(x), all_epoch_losses))
        plt.plot(avg_epoch_losses)
        plt.xlabel("epoch")
        plt.ylabel("avg rmse loss")
        plt.ylim((0,1))
        plt.title("Pipeline Regressor Average Training Loss")
        plt.savefig("avg_losses.png")
        plt.clf()

        plt.hist(losses, bins=20, range=(0,1))
        plt.savefig("loss_hist.png")
        plt.clf()


if __name__ == "__main__":
    main()
