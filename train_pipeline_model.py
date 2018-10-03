import json

import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from dataset import PipelineDataset, GroupDataLoader, load_data
from models import PrimitiveModel, AccuracyRegressionModel


primitive_submodel_dict = {}
input_model = nn.Module
output_model = nn.Module


# ensures that submodels exist for a given pipeline
def instantiate_submodels(pipeline, mf_len):
    primitive_names = pipeline.split("___")
    for primitive_name in primitive_names:
        if not primitive_name in primitive_submodel_dict:
            primitive_submodel_dict[primitive_name] = PrimitiveModel(mf_len)


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



def plot_losses(losses, save_dir="./plots/"):
    save_path = save_dir + "losses2.png"
    plt.plot(losses["train"], label="train")
    plt.plot(losses["test"], label="test")
    plt.legend(loc=0)
    plt.xlabel("Training Epoch")
    plt.ylabel("Average RMSE Loss")
    plt.yticks(np.arange(0, 70, 5) / 100)
    plt.title("Pipeline Regressor Loss")
    plt.savefig(save_path)
    plt.clf()


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    data = np.array(load_data("./data/processed_data.json"))
    folds = GroupDataLoader.cv_folds(data, "dataset", 10, seed=np.random.randint(2**32))
    train_data = data[folds[0][0]]
    test_data = data[folds[0][1]]
    train_dataloader = GroupDataLoader(
        data=train_data,
        group_label = "pipeline",
        dataset_class = PipelineDataset,
        dataset_params = {
            "feature_label": "metafeatures",
            "target_label": "test_accuracy",
            "device": "cuda:0"
        },
        batch_size = 48
    )
    test_dataloader = GroupDataLoader(
        data=test_data,
        group_label = "pipeline",
        dataset_class = PipelineDataset,
        dataset_params = {
            "feature_label": "metafeatures",
            "target_label": "test_accuracy",
            "device": "cuda:0"
        },
        batch_size = -1
    )

    input_size = next(iter(train_dataloader))[1][0].shape[1]

    global input_model
    input_model = PrimitiveModel(input_size)
    global output_model
    output_model = AccuracyRegressionModel(input_size)
    objective = torch.nn.MSELoss(reduction="elementwise_mean")

    losses = {
        "train": [],
        "test": []
    }
    n_epochs = 5000
    loop = tqdm(total=n_epochs, position=0)
    for e in range(n_epochs):
        epoch_losses = {
            "train": [],
            "test": []
        }

        for pipeline, (x_batch, y_batch) in train_dataloader:
            instantiate_submodels(pipeline, input_size)
            pipeline_regressor = build_model(pipeline, input_size)
            pipeline_regressor.cuda()

            optimizer = optim.Adam(pipeline_regressor.parameters(), lr=1e-4)
            optimizer.zero_grad()

            y_hat_batch = torch.squeeze(pipeline_regressor(x_batch))
            loss = torch.sqrt(objective(y_hat_batch, y_batch))

            loss.backward()
            optimizer.step()

            epoch_losses["train"].append(loss.item())

        for pipeline, (x_batch, y_batch) in test_dataloader:
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
            plot_losses(losses)
            json.dump(losses, open("losses2.json", "w"), indent=4)
            save_weights()

    loop.close()


if __name__ == "__main__":
    main()
