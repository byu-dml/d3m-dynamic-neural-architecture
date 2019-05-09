import numpy as np
import torch
import torch.optim as optim
import uuid

from problems.regression import Regression
from problems.siamese import Siamese
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
    loading_model = False
    task = "regression"

    if loading_model:
        # Set the name to a model that already exists
        id_ = '8c18d0e5-364d-4f9a-a61f-c9c0aeffdddf'
    else:
        # Create a new name for a new model to train
        id_ = uuid.uuid4()
    name = "{}_{}".format(task, id_)
    print('NAME:', name)

    seed = 1022357373
    n_epochs = 300
    batch_size = 32
    drop_last = True

    if task == "regression":
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
        problem_class = Regression

    elif task == "siamese":
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
        problem_class = Siamese

    problem = problem_class(
        batch_size = batch_size,
        drop_last = drop_last,
        seed = seed,
    )

    if loading_model:
        # Load the model
        print('Loading previous model')
        problem.model.load(config["weights_dir"])
    else:
        # Train a new model
        print('Training new model')
        print('Number Of Epochs:', n_epochs)
        learning_rate = 5e-5
        print('Learning Rate:', learning_rate)
        optimizer = optim.Adam(problem.model.parameters(), lr=learning_rate)  # Adam, SGD, Adagrad

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

    # Rank the pipelines using the model and compare to the true ranking using the spearman correlation coefficient
    training_SCC = problem.get_correlation_coefficient(problem.train_data_loader)
    validation_SCC = problem.get_correlation_coefficient(problem.validation_data_loader)
    print('Training Spearmann Correlation Coefficient:', training_SCC)
    print('Validation Spearmann Correlation Coefficient:', validation_SCC)
    # print("baselines", problem.baselines)

if __name__ == "__main__":
    main()
