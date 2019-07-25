import os
import json

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

from dna import utils
from dna.data import RNNDataLoader


class ModelBase:

    def __init__(self, *, seed):
        self.seed = seed
        self.fitted = False

    def fit(self, data, *, verbose=False):
        raise NotImplementedError()


class RegressionModelBase(ModelBase):

    def predict_regression(self, data, *, verbose=False):
        raise NotImplementedError()


class RankModelBase(ModelBase):

    def predict_rank(self, data, *, verbose=False):
        raise NotImplementedError()


class SubsetModelBase(ModelBase):

    def predict_subset(self, data, k, **kwargs):
        raise NotImplementedError()


class PyTorchModelBase:

    def __init__(self, *, y_dtype, device, seed):
        """
        Parameters
        ----------
        y_dtype:
            one of: torch.int64, torch.float32
        """
        self.y_dtype = y_dtype
        self.device = device
        self.seed = seed

        self._model = None

    def fit(
            self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
            verbose=False
    ):
        self._model = self._get_model(train_data)
        self._loss_function = self._get_loss_function()
        self._optimizer = self._get_optimizer(learning_rate)

        model_save_path = None
        if output_dir is not None:
            model_save_path = os.path.join(output_dir, 'model.pt')

        train_data_loader = self._get_data_loader(train_data, batch_size, drop_last, shuffle=True)
        validation_data_loader = None
        min_loss_score = np.inf
        if validation_data is not None:
            validation_data_loader = self._get_data_loader(validation_data, batch_size, drop_last=False, shuffle=False)

        for e in range(n_epochs):
            save_model = False
            if verbose:
                print('epoch {}'.format(e))

            self._train_epoch(
                train_data_loader, self._model, self._loss_function, self._optimizer, verbose=verbose
            )

            train_predictions, train_targets = self._predict_epoch(train_data_loader, self._model, verbose=verbose)
            train_loss_score = self._loss_function(train_predictions, train_targets)
            if output_dir is not None:
                self._save_outputs(output_dir, 'train', e, train_predictions, train_targets, train_loss_score)
            if verbose:
                print('train loss: {}'.format(train_loss_score))

            if validation_data_loader is not None:
                validation_predictions, validation_targets = self._predict_epoch(validation_data_loader, self._model, verbose=verbose)
                validation_loss_score = self._loss_function(validation_predictions, validation_targets)
                if output_dir is not None:
                    self._save_outputs(output_dir, 'validation', e, validation_predictions, validation_targets, validation_loss_score)
                if verbose:
                    print('validation loss: {}'.format(validation_loss_score))
                if validation_loss_score < min_loss_score:
                    min_loss_score = validation_loss_score
                    save_model = True
            else:
                if train_loss_score < min_loss_score:
                    min_loss_score = train_loss_score
                    save_model = True

            if save_model and model_save_path is not None:
                torch.save(self._model.state_dict(), model_save_path)

        if not save_model and model_save_path is not None:  # model not saved during final epoch
            self._model.load_state_dict(torch.load(model_save_path))

        self.fitted = True

    def _get_model(self, train_data):
        raise NotImplementedError()

    def _get_loss_function(self):
        raise NotImplementedError()

    def _get_optimizer(self, learning_rate):
        raise NotImplementedError()

    def _get_data_loader(self, data, batch_size, drop_last, shuffle):
        raise NotImplementedError()

    def _train_epoch(
            self, data_loader, model: nn.Module, loss_function, optimizer, *, verbose=True
    ):
        model.train()

        if verbose:
            progress = tqdm(total=len(data_loader), position=0)

        for x_batch, y_batch in data_loader:
            y_hat_batch = model(x_batch)
            loss = loss_function(y_hat_batch, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose:
                progress.update(1)

        if verbose:
            progress.close()

    def _predict_epoch(
            self, data_loader, model: nn.Module, *, verbose=True
    ):
        model.eval()
        predictions = []
        targets = []
        if verbose:
            progress = tqdm(total=len(data_loader), position=0)

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                y_hat_batch = model(x_batch)

                if y_batch.shape[0] == 1:
                    predictions.append(y_hat_batch.item())
                    targets.append(y_batch.item())
                else:
                    predictions.extend(y_hat_batch.tolist())
                    targets.extend(y_batch.tolist())

                if verbose:
                    progress.update(1)

        if verbose:
            progress.close()

        return torch.tensor(predictions, dtype=self.y_dtype), torch.tensor(targets, dtype=self.y_dtype)

    @staticmethod
    def _save_outputs(output_dir, phase, epoch, predictions, targets, loss_score):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_filename = phase + '_scores.csv'
        save_path = os.path.join(output_dir, save_filename)
        with open(save_path, 'a') as f:
            f.write(str(float(loss_score)) + '\n')

        output_dir = os.path.join(output_dir, 'outputs')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        save_filename = str(epoch) + '_' + phase + '.json'
        save_path = os.path.join(output_dir, save_filename)
        outputs = {
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
        }
        with open(save_path, 'w') as f:
            json.dump(outputs, f, separators=(',',':'))


class PyTorchRegressionRankSubsetModelBase(PyTorchModelBase, RegressionModelBase, RankModelBase, SubsetModelBase):

    def __init__(self, y_dtype, device, seed):
        # different arguments means different function calls
        PyTorchModelBase.__init__(self, y_dtype=torch.float32, device=device, seed=seed)
        RegressionModelBase.__init__(self, seed=seed)

    def predict_regression(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        data_loader = self._get_data_loader(data, batch_size, drop_last=False, shuffle=False)
        predictions, targets = self._predict_epoch(data_loader, self._model, verbose=verbose)
        reordered_predictions = predictions.numpy()[data_loader.get_group_ordering()]
        return reordered_predictions.tolist()

    def predict_rank(self, data, *, batch_size, verbose):
        if self._model is None:
            raise Exception('model not fit')

        predictions = self.predict_regression(data, batch_size=batch_size, verbose=verbose)
        ranks = utils.rank(predictions)
        return {
            'pipeline_id': [instance['pipeline_id'] for instance in data],
            'rank': ranks,
        }

    def predict_subset(self, data, k, *, batch_size, verbose=False):
        if self._model is None:
            raise Exception('model not fit')

        ranked_data = self.predict_rank(data, batch_size=batch_size, verbose=verbose)
        top_k = pd.DataFrame(ranked_data).nsmallest(k, columns='rank')['pipeline_id']
        return top_k.tolist()


class RNNRegressionRankSubsetModelBase(PyTorchRegressionRankSubsetModelBase):

    def __init__(self, device: str = 'cuda:0', seed: int = 0):

        super().__init__(y_dtype=torch.float32, seed=seed, device=device)

        self._data_loader_seed = seed + 1
        self._model_seed = seed + 2
        self.pipeline_structures = None
        self.num_primitives = None
        self.primitive_name_to_enc = None
        self.target_key = 'test_f1_macro'
        self.batch_group_key = 'pipeline_structure'
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'
        self.prim_name_key = 'name'
        self.prim_inputs_key = 'inputs'
        self.features_key = 'metafeatures'

    def _get_model(self, train_data):
        raise NotImplementedError()

    def fit(self, train_data, n_epochs, learning_rate, batch_size, drop_last, *, validation_data=None, output_dir=None,
            verbose=False):

        # Get the mapping of primitives to their one hot encoding
        self.primitive_name_to_enc = self._get_primitive_name_to_enc(train_data=train_data)

        PyTorchModelBase.fit(
            self, train_data, n_epochs, learning_rate, batch_size, drop_last, validation_data=validation_data,
            output_dir=output_dir, verbose=verbose
        )

    def _get_primitive_name_to_enc(self, train_data):
        primitive_names = set()

        # Get a set of all the primitives in the train set
        for instance in train_data:
            primitives = instance[self.pipeline_key][self.steps_key]
            for primitive in primitives:
                primitive_name = primitive[self.prim_name_key]
                primitive_names.add(primitive_name)

        # Get one hot encodings of all the primitives
        self.num_primitives = len(primitive_names)
        encoding = np.identity(n=self.num_primitives)

        # Create a mapping of primitive names to one hot encodings
        primitive_name_to_enc = {}
        primitive_names = sorted(primitive_names)
        for (primitive_name, primitive_encoding) in zip(primitive_names, encoding):
            primitive_name_to_enc[primitive_name] = primitive_encoding

        return primitive_name_to_enc

    def _get_loss_function(self):
        objective = torch.nn.MSELoss(reduction="mean")
        return lambda y_hat, y: torch.sqrt(objective(y_hat, y))

    def _get_optimizer(self, learning_rate):
        return torch.optim.Adam(self._model.parameters(), lr=learning_rate)

    def _get_data_loader(self, data, batch_size, drop_last, shuffle=True):
        return RNNDataLoader(
            data=data,
            group_key=self.batch_group_key,
            dataset_params={
                'features_key': self.features_key,
                'target_key': self.target_key,
                'y_dtype': self.y_dtype,
                'device': self.device
            },
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            seed=self._data_loader_seed,
            pipeline_structures=self.pipeline_structures,
            primitive_to_enc=self.primitive_name_to_enc,
            pipeline_key=self.pipeline_key,
            steps_key=self.steps_key,
            prim_name_key=self.prim_name_key
        )
