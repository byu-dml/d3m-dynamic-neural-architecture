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
    task = "siamese"
    seed = 1022357373
    n_epochs = 1000

    if task == "regression":
        name = "temp_regression"
        config = {
            "weights_dir": f"./results/{name}/weights",
            "outputs_dir": f"./results/{name}/outputs",
            "measure": rmse,
            "plot": {
                "train_label": "Train",
                "validation_label": "Validation",
                "title": "Regression Model RMSE",
                "ylabel": "RMSE",
                "path": f"./results/{name}/plot.pdf",
            }
        }
        problem = Regression(
           seed = seed,
        )
    elif task == "siamese":
        name = "temp_siamese"
        config = {
            "weights_dir": f"./results/{name}/weights",
            "outputs_dir": f"./results/{name}/outputs",
            "measure": accuracy,
            "plot": {
                "train_label": "Train",
                "validation_label": "Validation",
                "title": "Siamese Model Accuracy",
                "ylabel": "Accuracy",
                "path": f"./results/{name}/plot.pdf",
            }
        }
        problem = Siamese(
           seed = seed,
        )
    # problem.model.load(config["weights_dir"])
    learning_rate = 1e-4
    optimizer = optim.Adam(problem.model.parameters(), lr=learning_rate)

    trainer = PyTorchModelTrainer(
        problem.model,
        problem.train_data_loader,
        problem.validation_data_loader,
        problem.test_data_loader,
        problem.loss_function
    )

    train_measurements = []
    validation_measurements = []
    for e in range(n_epochs):
        train_results = trainer.train_epoch(optimizer)
        problem.model.save(config["weights_dir"])
        trainer.save_outputs(config["outputs_dir"], n_epochs)

        train_predictions = train_results[0]
        train_targets = train_results[1]
        validation_predictions = train_results[2]
        validation_targets = train_results[3]

        train_measurements.append(config["measure"](
            train_predictions, train_targets
        ))
        validation_measurements.append(config["measure"](
            validation_predictions, validation_targets
        ))
        print(
            "train {} validation {}".format(
                round(train_measurements[-1], 4),
                round(validation_measurements[-1], 4)
            )
        )

        trainer.plot(
            {
                config["plot"]["train_label"]: train_measurements,
                config["plot"]["validation_label"]: validation_measurements
            },
            config["plot"]["title"],
            config["plot"]["ylabel"],
            path = config["plot"]["path"]
        )
    print("baselines", problem.baselines)

if __name__ == "__main__":
    main()
