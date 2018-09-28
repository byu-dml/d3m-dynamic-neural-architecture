import json

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch


class PipelineDataset(Dataset):

    def __init__(self, path, device):
        self.data = json.load(open(path, "r"))
        self.device = device

    # decides what to do with inside of the brackets (e.g. dataset[0])
    def __getitem__(self, item):
        # other processing can be done here
        x = torch.tensor(self.data[item]["metafeatures"], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data[item]["test_accuracy"], dtype=torch.float32, device=self.device)
        pipeline = self.data[item]["job_str"].split("___")[1:]
        return x, y, pipeline

    # defines how big the dataset is. Return a smaller value (either some fraction or a hard coded number of
    # data points) if you want to get a train dataset.
    def __len__(self):
        return len(self.data)


def preprocess_data(path):
    data = json.load(open(path, "r"))
    metafeatures = []
    for pipeline in data:
        metafeatures.append(pipeline["metafeatures"])
    mf_df = pd.DataFrame(metafeatures)
    non_time_cols = [col for col in mf_df.columns if not "time" in col.lower()]
    mf_df = mf_df[non_time_cols]
    processed_mf_df = mf_df.dropna(axis=1, how="any")
    processed_data = []
    for pipeline, mfs in zip(data, processed_mf_df.values.tolist()):
        processed_data.append({
            "job_str": pipeline["job_str"],
            "metafeatures": mfs,
            "test_accuracy": pipeline["test_accuracy"]
        })
    json.dump(processed_data, open("./data/processed_data.json", "w"), indent=4)

