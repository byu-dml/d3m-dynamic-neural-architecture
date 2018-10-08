import json

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

from models import PrimitiveModel, RegressionModel, ClassificationModel, SiameseModel

from problems import Siamese, Regression


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


def plot_losses(losses, baseline_losses, name=""):
    save_dir = "./results/plots"
    save_path = os.path.join(save_dir, name + ".png")
    plt.plot(losses["train"], label="train")
    plt.plot(losses["test"], label="test")
    for label, loss in baseline_losses.items():
        plt.plot([loss] * len(losses["train"]), label=label)
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
    id_ = "test"
    print(id_)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    problem = Siamese(
        drop_last = True, # workaround todo handle batches of size 1
        device = device
    )

    losses = {
        "train": [],
        "test": []
    }
    n_epochs = 500
    learning_rate = 1e-3
    for e in range(n_epochs):
        loop = tqdm(total=len(problem.train_data_loader), position=0)
        epoch_losses = {
            "train": [],
            "test": []
        }
        for pipelines, (x_batch, y_batch) in problem.train_data_loader:
            model = problem.model
            model.cuda()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            optimizer.zero_grad()

            y_hat_batch = torch.squeeze(model(x_batch, *pipelines))
            loss = problem.loss_function(y_hat_batch, y_batch)

            loss.backward()
            optimizer.step()

            epoch_losses["train"].append(loss.item())
            loop.set_description(
                "train_loss: {:.4f}".format(epoch_losses["train"][-1])
            )
            loop.update(1)

        loop.close()
        losses["train"].append(np.average(epoch_losses["train"]))

        loop = tqdm(total=len(problem.test_data_loader), position=0)
        for pipelines, (x_batch, y_batch) in problem.test_data_loader:
            model = problem.model
            model.cuda()

            y_hat_batch = torch.squeeze(model(x_batch, *pipelines))
            loss = problem.loss_function(y_hat_batch, y_batch)

            epoch_losses["test"].append(loss.item())
            loop.set_description(
                "train_loss: {:.4f}".format(epoch_losses["test"][-1])
            )
            loop.update(1)
        loop.close()


        losses["test"].append(np.average(epoch_losses["test"]))
        print("train loss: {} test loss: {}".format(losses["train"][-1], losses["test"][-1]))

        if (e+1) % 1 == 0:
            plot_losses(losses, problem.baseline_losses, id_+"_losses")
            json.dump(losses, open(f"{id_}_losses.json", "w"), indent=4)
            # save_weights()




if __name__ == "__main__":
    main()
