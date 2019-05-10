import random
from typing import List, Dict

from torch.utils.data import Sampler, DataLoader

from .regression_dataset import RegressionDataset
from data import group_json_objects


class RandomSampler(Sampler):
    """
    Samples indices uniformly without replacement.

    Parameters
    ----------
    n: int
        the number of indices to sample
    seed: int
        used to reproduce randomization
    """

    def __init__(self, n, seed):
        self.n = n
        self._indices = list(range(n))
        self._random = random.Random()
        self._random.seed(seed)

    def __iter__(self):
        self._random.shuffle(self._indices)
        return iter(self._indices)

    def __len__(self):
        return self.n


class RegressionDataLoader(object):
    """
    Batches a dataset for PyTorch Neural Network training. Partitions the
    dataset so that batches belong to the same group.

    Parameters:
    -----------
    data: List[Dict], JSON compatible list of objects representing a dataset.
        dataset_class must know how to parse the data given dataset_params.
    group_key: str, pipeline run data is grouped by group_key and each
        batch of data comes from only one group. group_key must be a key into
        each element of the pipeline run data. the value of group_key must be
        hashable.
    dataset_class: Type[torch.utils.data.Dataset], the class used to make
        dataset instances after the dataset is partitioned.
    dataset_params: dict, extra parameters needed to instantiate dataset_class
    batch_size: int, the number of data points in each batch
    drop_last: bool, default False. whether to drop the last incomplete batch.
    shuffle: bool, default True. whether to randomize the batches.
    """

    def __init__(
        self, data: List[Dict], group_key: str, pipeline_key: str, dataset_params: dict,
            batch_size: int, drop_last: bool, shuffle: bool,
        seed: int
    ):
        self.data = data
        self.group_key = group_key
        self.pipeline_key = pipeline_key
        self.dataset_params = dataset_params
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self._random = random.Random()
        self._random.seed(seed)

        self._init_dataloaders()
        self._init_group_metadataloader()

    def _init_dataloaders(self):
        """
        Groups self.data based on group_key. Creates a
        torch.utils.data.DataLoader for each group, using self.dataset_class.
        """
        # group the data
        grouped_data = group_json_objects(self.data, self.group_key)

        # create dataloaders
        self._group_dataloaders = {}
        for group, group_indices in grouped_data.items():
            group_data = [self.data[i] for i in group_indices]
            group_dataset = RegressionDataset(group_data, **self.dataset_params)
            new_dataloader = self._get_data_loader(
                group_dataset
            )
            # assert(len(new_dataloader) != 0)
            self._group_dataloaders[group] = new_dataloader

    def _get_data_loader(self, data):
        if self.shuffle:
            sampler = RandomSampler(len(data), self._randint())
        else:
            sampler = None
        dataloader = DataLoader(
            dataset = data,
            sampler =  sampler,
            batch_size = self.batch_size,
            drop_last = self.drop_last
        )
        # assert(len(dataloader) != 0)
        return dataloader

    def _randint(self):
        return self._random.randint(0,2**32-1)

    def _init_group_metadataloader(self):
        """
        Creates a dataloader which randomizes the batches over the groups. This
        allows the order of the batches to be independent of the groups.
        """
        self._group_batches = []
        for group, group_dataloader in self._group_dataloaders.items():
            # assert(len(group_dataloader) != 0)
            self._group_batches += [group] * len(group_dataloader)
        self._random.shuffle(self._group_batches)

    def __iter__(self):
        return iter(self._iter())

    def _iter(self):
        group_dataloader_iters = {}
        for group in self._group_batches:
            if not group in group_dataloader_iters:
                group_dataloader_iters[group] = iter(
                    self._group_dataloaders[group]
                )
            # Get the next batch of metafeatures vectors, targets, and their corresponding data set ids
            (dataset_ids, x_batch), y_batch = next(group_dataloader_iters[group])

            # Since all pipeline are the same in this group, just grab one of them
            pipeline = self._group_dataloaders[group].dataset.data[0][self.pipeline_key]
            yield (group, pipeline, x_batch, dataset_ids), y_batch
        raise StopIteration()

    def __len__(self):
        return len(self._group_batches)
