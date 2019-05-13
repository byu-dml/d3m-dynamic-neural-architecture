import numpy as np
import torch
import torch.optim as optim
import uuid

from problems.regression import Regression
from problems.siamese import Siamese
from pytorch_model_trainer import PyTorchModelTrainer

from problems.autosklearn_metalearner import AutoSklearnMetalearner


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
    n_epochs = 0
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
        # print("baselines", problem.baselines)

    k = 50
    use_test = False
    dataset_performances_train = problem.dataloader_to_map(problem.train_data_loader)
    # for brandon -> why is this test_data_loader.  Validation dataloader is the same as train for some reason
    dataset_performances_validate = problem.dataloader_to_map(problem.test_data_loader)

    # Rank the pipelines using the model and compare to the true ranking using the spearman correlation coefficient
    training_SCC, top_k = problem.get_correlation_coefficient(dataset_performances_train, k)
    validation_SCC, top_k_valid = problem.get_correlation_coefficient(dataset_performances_validate, k)
    print('Training Spearmann Correlation Coefficient:', training_SCC)
    print('Validation Spearmann Correlation Coefficient:', validation_SCC)

    if task == "regression":
        # the metric stays test accuracy because this is the name of the metric in our metadata file, however if that is fixed this should change
        metric = 'test_accuracy'
        maximize_metric = False
    else:
        metric = 'test_accuracy'
        maximize_metric = True

    data_to_pass = list(dataset_performances_validate.keys()) if not use_test else None
    metalearner = AutoSklearnMetalearner(data_to_pass, metric=metric, maximize_metric=maximize_metric,
                                         use_test=data_to_pass is None)
    metric_differences, top_pipeline_values, top_k_out_of_total, top_pipelines_per_dataset = metalearner.get_metric_difference_from_best(k)
    mean_difference = np.mean(list(metric_differences.values()))
    print(mean_difference)








if __name__ == "__main__":
    main()
