from typing import List, Dict
from torch.utils.data import Dataset
import torch


class RegressionDataset(Dataset):
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
        self, data: List[Dict], data_set_key, features_key: str, target_key: str, device: str
    ):
        self.data = data
        self.features_key = features_key
        self.target_key = target_key
        self.dataset_key = data_set_key
        self.device = device
        self._y_dtype = torch.float32

    def __getitem__(self, index: int):
        item = self.data[index]
        dataset = item[self.dataset_key]
        metafeatures = torch.tensor(
            item[self.features_key],
            dtype=torch.float32,
            device=self.device
        )
        x = (dataset, metafeatures)
        y = torch.tensor(
            item[self.target_key],
            dtype=self._y_dtype,
            device=self.device
        )
        return x, y

    def __len__(self):
        return len(self.data)

