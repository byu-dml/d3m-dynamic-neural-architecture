import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

from data import write_json
from problems import Siamese, Regression
from pytorch_model_trainer import PyTorchModelTrainer


def save_weights():
	for key, model in primitive_submodel_dict.items():
		torch.save(model, "%s.pt" % key)


def accuracy(y_hat, y):
    y_hat = np.argmax(y_hat, axis=1)
    return np.sum(y_hat == y, dtype=np.float32) / len(y)


def rmse(y_hat, y):
    return np.average((np.array(y_hat) - np.array(y))**2)**.5


def main():
    task = "regression"
    if task == "regression":
        name = "temp_Regression_small_no_cleaning_modeling"
        config = {
            "plot_args": {
                "measures": {"RMSE": rmse},
                "title": "Regression Model RMSE",
                "ylabel": "RMSE",
                "path": f"./results/plots/{name}.png"
            },
            "results_path": f"./results/{name}.json"
        }
        problem = Regression(
           seed = 3216827855,
        )
    elif task == "siamese":
        name = "temp_siamese"
        config = {
            "plot_args": {
                "measures": {"Accuracy": accuracy},
                "title": "Siamese Model Accuracy",
                "ylabel": "RMSE",
                "path": f"./results/plots/{name}.png"
            },
            "results_path": f"./results/{name}.json"
        }
        problem = Siamese(
           seed = 37,
        )

    learning_rate = 1e-4
    n_epochs = 500
    optimizer = optim.Adam(problem.model.parameters(), lr=learning_rate)
    trainer = PyTorchModelTrainer(
        problem.model,
        problem.train_data_loader,
        problem.validation_data_loader,
        problem.test_data_loader,
        problem.loss_function
    )
    for e in range(n_epochs):
        trainer.train_epoch(optimizer)
        trainer.plot_results(**config["plot_args"])
    trainer.save_results(config["results_path"])

if __name__ == "__main__":
    main()
