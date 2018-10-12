import json

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import os

from models import PrimitiveModel, RegressionModel, ClassificationModel, SiameseModel
from pytorch_model_trainer import PyTorchModelTrainer
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


def accuracy(y_hat, y):
    y_hat = torch.argmax(y_hat, dim=1)

    return torch.sum(y_hat == y, dtype=torch.float32) / y.shape[0]


def main2():
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
    accuracies = {
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
        epoch_accuracies = {
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
            epoch_accuracies["train"].append(accuracy(y_hat_batch, y_batch))
            loss.backward()
            optimizer.step()

            epoch_losses["train"].append(loss.item())
            loop.update(1)

        loop.close()
        losses["train"].append(np.average(epoch_losses["train"]))
        accuracies["train"].append(np.average(epoch_accuracies["train"]))

        loop = tqdm(total=len(problem.test_data_loader), position=0)
        with torch.no_grad(): # don't compute gradients
            for pipelines, (x_batch, y_batch) in problem.test_data_loader:
                model = problem.model
                model.cuda()

                y_hat_batch = torch.squeeze(model(x_batch, *pipelines))
                loss = problem.loss_function(y_hat_batch, y_batch)
                epoch_accuracies["test"].append(accuracy(y_hat_batch, y_batch))
                epoch_losses["test"].append(loss.item())
                loop.update(1)
        loop.close()


        losses["test"].append(np.average(epoch_losses["test"]))
        accuracies["test"].append(np.average(epoch_accuracies["test"]))
        print("train acc: {} test acc: {}".format(accuracies["train"][-1], accuracies["test"][-1]))

        if (e+1) % 1 == 0:
            plot_losses(accuracies, problem.baseline_losses, id_+"_losses")
            json.dump(losses, open(f"{id_}_losses.json", "w"), indent=4)
            # save_weights()
    

def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)

    return np.sum(y_hat == y, dtype=np.float32) / len(y)

def main():
    use_cuda = torch.cuda.is_available()
    problem = Siamese(
        drop_last = True, # workaround todo handle batches of size 1
        device = torch.device("cuda:0" if use_cuda else "cpu")
    )
    problem.model.cuda()
    learning_rate = 1e-4
    optimizer = optim.Adam(problem.model.parameters(), lr=learning_rate)
    pmt = PyTorchModelTrainer(
        problem.model,
        problem.train_data_loader,
        problem.test_data_loader,
        None,
        problem.loss_function
    )
    pmt.train_epoch(optimizer)
    pmt.train_epoch(optimizer)
    pmt.plot_results(
        {"Accuracy": accuracy},
        "Siamese Model Accuracy",
        "Accuracy"
    )

if __name__ == "__main__":
    main()
