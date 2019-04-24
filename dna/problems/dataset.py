from typing import List, Dict

import torch
from torch.utils.data import Dataset


class Dataset(Dataset):
    """
    A subclass of torch.utils.data.Dataset for handling simple JSON structed
    data.

    Parameters:
    -----------
    data: List[Dict], JSON structed data.
    features_key: str, the key into each element of data whose value is a list
        of features used for input to a PyTorch network.
    target_key: str, the key into each element of data whose value is the
        target used for a PyTorch network.
    device": str, the device onto which the data will be loaded
    """

    def __init__(
        self, data: List[Dict], features_key: str, target_key: str,
        task_type: str, device: str
    ):
        self.data = data
        self.features_key = features_key
        self.target_key = target_key
        self.task_type = task_type
        if self.task_type == "CLASSIFICATION":
            self._y_dtype = torch.int64
        elif self.task_type == "REGRESSION":
            self._y_dtype = torch.float32
        self.device = device

    def __getitem__(self, item: int):
        x = torch.tensor(
            self.data[item][self.features_key],
            dtype=torch.float32,
            device=self.device
        ),
        y = torch.tensor(
            self.data[item][self.target_key],
            dtype=self._y_dtype,
            device=self.device
        )
        return x, y

    def __len__(self):
        return len(self.data)
