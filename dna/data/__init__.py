import os
import json
import tarfile
import random
from typing import List, Dict

import numpy as np
import pandas as pd


DATA_DIR = "./dna/data"
RAW_DATA_NAME = "complete_pipelines_and_metafeatures"
COMPRESSED_RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME + ".tar.xz")
RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME + ".json")
ALL_DATA_PATH = os.path.join(DATA_DIR, "all_data.json")
TRAIN_TEST_SPLIT_SEED = 1607242652
N_TEST_DATASETS = 42
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_data.json")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test_data.json")


def read_json(path):
    """
    Reads JSON formatted data.
    """
    return json.load(open(path, "r"))

def write_json(data, path, pretty=False):
    """
    Writes JSON structured data. Indents 4 spaces when pretty is True.
    """
    if pretty:
        json.dump(data, open(path, "w"), indent=4, sort_keys=True)
    else:
        json.dump(data, open(path, "w"))

def extract_data():
    with tarfile.open(COMPRESSED_RAW_DATA_PATH, "r:xz") as tar:
        tar.extract(RAW_DATA_NAME + ".json", DATA_DIR)

def reformat_data():
    data = read_json(RAW_DATA_PATH)
    reformatted_data = []
    for item in data:
        dataset, pipeline = item["job_str"].split("___", 1)
        metafeatures = {
            k: v for k, v in item["metafeatures"].items() if not "time" in k.lower()
        }
        metafeatures["TotalTime"] = item["metafeatures_time"]
        train_time = item["train_fit_time"] + item["train_predict_time"]
        reformatted_data.append({
            "dataset": dataset,
            "pipeline": pipeline,
            "metafeatures": metafeatures,
            "train_accuracy": item["train_accuracy"],
            "test_accuracy": item["test_accuracy"],
            "train_time": train_time,
            "test_time": item["test_predict_time"]
        })
    write_json(reformatted_data, ALL_DATA_PATH, pretty=True)

def group_json_objects(json_objects, group_key):
    """
    Groups JSON data by group_key.

    Parameters:
    -----------
    json_objects: List[Dict], JSON compatible list of objects.
    group_key: str, json_objects is grouped by group_key. group_key must be a
        key into each object in json_objects and the corresponding value must
        be hashable.

    Returns:
    --------
    A dict with key being a group and the value is a list of indices into
    json_objects.
    """
    grouped_objects = {}
    for i, obj in enumerate(json_objects):
        group = obj[group_key]
        if not group in grouped_objects:
            grouped_objects[group] = []
        grouped_objects[group].append(i)
    return grouped_objects

def drop_nan_metafeatures():
    all_data = read_json(ALL_DATA_PATH)

    mfs = pd.DataFrame([item["metafeatures"] for item in all_data])
    mfs.replace(
        to_replace=[np.inf, - np.inf], value=np.nan, inplace=True
    )

    drop_cols = list(mfs.columns[mfs.isnull().any()])
    mfs.drop(labels=drop_cols, axis=1, inplace=True)

    for item, mfs in zip(all_data, mfs.values.tolist()):
        item["metafeatures"] = mfs

    write_json(all_data, ALL_DATA_PATH, pretty=True)

def split_data():
    all_data = read_json(ALL_DATA_PATH)
    grouped_data_indices = group_json_objects(all_data, "dataset")
    groups = list(grouped_data_indices.keys())

    rnd = random.Random()
    rnd.seed(TRAIN_TEST_SPLIT_SEED)
    rnd.shuffle(groups)

    train_data = []
    for group in groups[N_TEST_DATASETS:]:
        for i in grouped_data_indices[group]:
            train_data.append(all_data[i])

    test_data = []
    for group in groups[:N_TEST_DATASETS]:
        for i in grouped_data_indices[group]:
            test_data.append(all_data[i])

    write_json(train_data, TRAIN_DATA_PATH, pretty=True)
    write_json(test_data, TEST_DATA_PATH, pretty=True)

def make_cv_folds(
    data: List[Dict], group_key: str, n_folds: int = -1, seed: int = 0
):
    """
    Generates cross validation folds with indices into data. Places data points
    belonging to the same group into the same train/test portion of each fold.
    """
    grouped_data_indices = group_json_objects(data, group_key)
    if n_folds < 0:
        n_folds = len(grouped_data_indices)

    groups = list(grouped_data_indices.keys())
    group_type = type(groups[0])
    rnd = random.Random()
    rnd.seed(seed)
    rnd.shuffle(groups)

    split_groups = np.array_split(groups, n_folds)

    folds = []
    for i in range(n_folds):
        train_indices = []
        test_indices = []
        for j, split_group in enumerate(split_groups):
            for group in split_group:
                group = group_type(group)
                indices = grouped_data_indices[group]
                if i == j:
                    test_indices += indices
                else:
                    train_indices += indices
        folds.append((train_indices, test_indices))

    return folds


def main():
    extract_data()
    reformat_data()
    drop_nan_metafeatures()
    split_data()

if __name__ == '__main__':
    main()
