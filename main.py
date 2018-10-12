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


def main():
    use_cuda = torch.cuda.is_available()
    problem = Siamese(
        seed = 37
    )
    learning_rate = 1e-4
    n_epochs = 50
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
        trainer.plot_results(
            {"Accuracy": accuracy},
            "Siamese Model Accuracy",
            "Accuracy",
            path="./results/plots/siamese.png"
        )
        if (e + 1) % 3 == 0:
            trainer.save_results("./results/siamese.json")

if __name__ == "__main__":
    main()
