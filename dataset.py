import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torchvision import transforms, utils, datasets

class pipelineDataset(Dataset):
    def __init__(self, root, train=True):
        self.data = json.load(open(root, "r"))

    # decides what to do with inside of the brackets (e.g. dataset[0])
    def __getitem__(self, item):
        print(item)
        # other processing can be done here
        return self.data[item]

    # defines how big the dataset is. Return a smaller value (either some fraction or a hard coded number of
    # data points) if you want to get a train dataset.
    def __len__(self):
        return len(self.data)