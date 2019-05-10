import os

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

        self._n_completed_epochs = 0

        self._train_predictions = None
        self._train_targets = None
        self._validation_predictions = None
        self._validation_targets = None
        self._test_predictions = None
        self._test_targets = None

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
        progress.set_description("epoch {}".format(
            self._n_completed_epochs + 1
        ))
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
        self._train_predictions, self._train_targets = self._epoch(
            self.train_data_loader, optimizer
        )
        self._validation_predictions, self._validation_targets = self._epoch(
            self.validation_data_loader
        )

        self._n_completed_epochs += 1

        return (
            self._train_predictions, self._train_targets,
            self._validation_predictions, self._validation_targets
        )

    def test(self):
        self._test_predictions, self._test_targets = self._epoch(self.test_data_loader)
        return self._test_predictions, self._test_targets

    @classmethod
    def plot(
        cls, measures, title = None, ylabel = None, xlabel = "Epoch",
        path = None
    ):
        """
        Computes each measure on the train and validation data per epoch and
        plots the measures over epochs. Uses the keys in measures as the labels
        in the plot legend.

        Parameters
        ----------
        measures: Dict[str, List[float]]
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
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.yaxis.set_ticks_position('both')

        for label, measure in measures.items():
            plt.plot(measure, label = label)

        plt.legend(loc=0)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
        plt.clf()
        plt.close()

    def save_outputs(self, save_dir, n_total_epochs = None):
        # todo expose inner method to avoid test data epoch problem
        if n_total_epochs is None:
            n_total_epochs = 0

        epoch_str = str(self._n_completed_epochs).zfill(
            len(str(n_total_epochs))
        )

        self._save_outputs(
            self._train_predictions,
            self._train_targets,
            "train",
            epoch_str,
            save_dir
        )
        self._save_outputs(
            self._validation_predictions,
            self._validation_targets,
            "validation",
            epoch_str,
            save_dir
        )
        self._save_outputs(
            self._test_predictions,
            self._test_targets,
            "validation",
            epoch_str,
            save_dir
        )

    def _save_outputs(self, predictions, targets, phase, epoch_str, save_dir):
        if not (predictions is None or targets is None):
            save_dir = os.path.join(save_dir, phase)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            path = os.path.join(save_dir, f"{epoch_str}.json")
            results = {
                "phase": phase,
                "epoch": epoch_str,
                "predictions": predictions,
                "targets": targets
            }
            write_json(results, path)

    @property
    def n_completed_epochs(self):
        return self._n_completed_epochs

    @property
    def train_predictions(self):
        return self._train_predictions

    @property
    def validation_predictions(self):
        return self._validation_predictions

    @property
    def test_predictions(self):
        return self._test_predictions
