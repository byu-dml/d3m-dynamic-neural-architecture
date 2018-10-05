import json

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

from models import PrimitiveModel, RegressionModel, ClassificationModel

from problems import Problem


primitive_submodel_dict = {}
input_model = nn.Module
output_model = nn.Module


# ensures that submodels exist for a given pipeline
def instantiate_submodels(pipeline, mf_len):
    primitive_names = pipeline.split("___")
    for primitive_name in primitive_names:
        if not primitive_name in primitive_submodel_dict:
            primitive_submodel_dict[primitive_name] = PrimitiveModel(primitive_name, mf_len)


# Dynamically builds model for a given pipeline
# pipeline is a string of primitive names, each separated by "___"
def build_model(pipeline, mf_len):
    primitive_names = pipeline.split("___")
    primitive_models = []
    # Prepend an input model
    primitive_models.append(input_model)
    # Add all of the primitive models in order
    for primitive_name in primitive_names:
        primitive_models.append(primitive_submodel_dict[primitive_name])
    # Append output model to the end of the pipeline model
    primitive_models.append(output_model)

    return nn.Sequential(*primitive_models)


def save_weights():
	for key, model in primitive_submodel_dict.items():
		torch.save(model, "%s.pt" % key)


def plot_losses(losses, name=""):
    save_dir = "./results/plots"
    save_path = os.path.join(save_dir, name + ".png")
    plt.plot(losses["train"], label="train")
    plt.plot(losses["test"], label="test")
    plt.legend(loc=0)
    plt.xlabel("Training Epoch")
    plt.ylabel("Average RMSE Loss")
    plt.yticks(np.arange(0, 1.05, .05))
    plt.title("Pipeline Regressor Loss")
    plt.savefig(save_path)
    plt.clf()


class DNAExperimenter(object):
    """
    A Dynamic Neural Architecture experiment using the PyTorch framework
    """

    def __init__(self, exp_name, train_data, test_data):
        self.exp_name = exp_name
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )




def main():
    id_ = "3_layer_with_single_batch_norm_except_on_task_model"
    print(id_)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    problem = Problem(
        device = device
    )

    input_size = problem.feature_size
    output_size = problem.target_size

    global input_model
    input_model = PrimitiveModel("input", input_size)
    global output_model
    output_model = RegressionModel(input_size)
    # objective = torch.nn.MultiLabelSoftMarginLoss(reduction="elementwise_mean")
    objective = torch.nn.MSELoss(reduction="elementwise_mean")

    losses = {
        "train": [],
        "test": []
    }
    n_epochs = 5000
    loop = tqdm(total=n_epochs, position=0)
    learning_rate = 1e-4
    for e in range(n_epochs):
        epoch_losses = {
            "train": [],
            "test": []
        }
        for pipeline, (x_batch, y_batch) in problem.train_data_loader:
            instantiate_submodels(pipeline, input_size)
            pipeline_regressor = build_model(pipeline, input_size)
            pipeline_regressor.cuda()

            optimizer = optim.Adam(pipeline_regressor.parameters(), lr=learning_rate)
            optimizer.zero_grad()

            y_hat_batch = torch.squeeze(pipeline_regressor(x_batch))
            loss = torch.sqrt(objective(y_hat_batch, y_batch))

            loss.backward()
            optimizer.step()

            epoch_losses["train"].append(loss.item())

        for pipeline, (x_batch, y_batch) in problem.test_data_loader:
            instantiate_submodels(pipeline, input_size)
            pipeline_regressor = build_model(pipeline, input_size)
            pipeline_regressor.cuda()

            y_hat_batch = torch.squeeze(pipeline_regressor(x_batch))
            loss = torch.sqrt(objective(y_hat_batch, y_batch))

            epoch_losses["test"].append(loss.item())

        losses["train"].append(np.average(epoch_losses["train"]))
        losses["test"].append(np.average(epoch_losses["test"]))
        loop.set_description(
            "train_loss: {:.4f} test_loss: {:.4f}".format(
                losses["train"][-1], losses["test"][-1]
            )
        )
        loop.update(1)
        if (e+1) % 10 == 0:
            plot_losses(losses, id_+"_losses")
            json.dump(losses, open(f"{id_}_losses.json", "w"), indent=4)
            # save_weights()

    loop.close()


if __name__ == "__main__":
    main()
