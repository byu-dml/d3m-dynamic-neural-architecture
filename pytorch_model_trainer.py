import json

from tqdm import tqdm
from matplotlib import pyplot as plt
import torch

from data import write_json

try:
    from contextlib import nullcontext
except ImportError:
    class nullcontext(object):
        def __init__(self, dummy_resource=None):
            self.dummy_resource = dummy_resource
        def __enter__(self):
            return self.dummy_resource
        def __exit__(self, *args):
            pass


class PyTorchModelTrainer(object):

    def __init__(
        self, model, train_data_loader, validation_data_loader,
        test_data_loader, loss_f
    ):
        self.model = model
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.test_data_loader = test_data_loader
        self.loss_f = loss_f

        # each element in the these list represents an entire epoch's data
        self._results = {
            "train": {
                "predictions": [],
                "targets": []
            },
            "validation": {
                "predictions": [],
                "targets": []
            },
            "test": {
                "predictions": [],
                "targets": []
            }
        }

    def _epoch(self, data_loader, optimizer=None):
        if optimizer is None:
            self.model.eval()
            context_manager = torch.no_grad()
        else:
            self.model.train()
            context_manager = nullcontext()

        predictions = []
        targets = []
        progress = tqdm(
            total = len(data_loader),
            position = 0
        )

        with context_manager:
            for x_batch, y_batch in data_loader:
                y_hat_batch = self.model(x_batch)

                if not optimizer is None:
                    loss = self.loss_f(y_hat_batch, y_batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                predictions.extend(y_hat_batch.tolist())
                targets.extend(y_batch.tolist())
                progress.update(1)

        progress.close()

        return predictions, targets

    def train_epoch(self, optimizer):
        # todo in case of failure, reset model to before epoch
        train_predictions, train_targets = self._epoch(
            self.train_data_loader, optimizer
        )
        validation_predictions, validation_targets = self._epoch(
            self.validation_data_loader
        )
        # record data only after an entire epoch has completed
        self._results["train"]["predictions"].append(train_predictions)
        self._results["train"]["targets"].append(train_targets)
        self._results["validation"]["predictions"].append(validation_predictions)
        self._results["validation"]["targets"].append(validation_targets)

    def test(self):
        if len(self._results["test"]["predictions"]) > 0:
            raise Exception("Evaluating test data can only be performed once")

        predictions, targets = self._epoch(self.test_data_loader)
        self._results["test"]["predictions"] = predictions
        self._results["test"]["targets"] = targets

    def plot_results(self, measures, title, ylabel=None, xlabel="Epoch", path=None):
        """
        Computes each measure on the train and validation data per epoch and
        plots the measures over epochs. Uses the keys in measures as the labels
        in the plot legend.

        Parameters
        ----------
        measures: Dict[str, Callable[Iterable, Iterable]]
            A mapping of strings to functions. Each function performs some
            measure of model performance using the predictions and targets on
            a per epoch basis. The string key is the text for the label in the
            plot legend and will be prefixed with "Train " and "Validation "
            for the respective data.
        title: str
            The title for the plot.
        ylabel: str
            The label for the y axis.
        xlabel: str, default - "epoch"
            The label for the x axis
        path: str, default None
            The path in which to save the plot. If None, then the plot is shown
            instead.
        """

        for label, measure in measures.items():
            measured_train_data = []
            for predictions, targets in zip(
                self._results["train"]["predictions"],
                self._results["train"]["targets"]
            ):
                measured_train_data.append(measure(predictions, targets))

            measured_validation_data = []
            for predictions, targets in zip(
                self._results["validation"]["predictions"],
                self._results["validation"]["targets"]
            ):
                measured_validation_data.append(measure(predictions, targets))

            x = range(1, len(measured_train_data) + 1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.yaxis.set_ticks_position('both')
            plt.plot(x, measured_train_data, label=f"Train {label}")
            plt.plot(x, measured_validation_data, label=f"Validation {label}")
            # todo baselines
            plt.legend(loc=0)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            if path is None:
                plt.show()
            else:
                plt.savefig(path)
            plt.clf()

    def save_results(self, path):
        write_json(self._results, path, pretty=True)

    @property
    def n_completed_epochs(self):
        return len(self._results["train"]["predictions"])

    @property
    def train_predictions(self):
        return self._train_predictions

    @property
    def validation_predictions(self):
        return self._validation_predictions

    @property
    def test_predictions(self):
        return self._test_predictions
