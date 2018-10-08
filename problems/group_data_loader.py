import random
from typing import Type, List, Dict

from torch.utils.data import DataLoader

from data import group_json_objects
from problems import Dataset

class GroupDataLoader(object):
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
        self, data: List[Dict], group_key: str, dataset_class: Type[Dataset],
        dataset_params: dict, batch_size: int, drop_last: bool = False,
        shuffle: bool = True
    ):
        self.data = data
        self.group_key = group_key
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        if batch_size == -1:
            batch_size = len(self.data)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

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
            group_dataset = self.dataset_class(group_data, **self.dataset_params)
            self._group_dataloaders[group] = DataLoader(
                dataset=group_dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                drop_last=self.drop_last
            )

    def _init_group_metadataloader(self):
        """
        Creates a dataloader which randomizes the batches over the groups. This
        allows the order of the batches to be independent of the groups.
        """
        self._group_batches = []
        for group, group_dataloader in self._group_dataloaders.items():
            self._group_batches += [group] * len(group_dataloader)
        random.shuffle(self._group_batches)

    def __iter__(self):
        group_dataloader_iters = {}
        for group in self._group_batches:
            if not group in group_dataloader_iters:
                group_dataloader_iters[group] = iter(
                    self._group_dataloaders[group]
                )
            yield group, next(group_dataloader_iters[group])
        raise StopIteration()

    def __len__(self):
        return len(self._group_batches)
