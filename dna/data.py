import json
import os
import random
import tarfile
import typing


DATA_DIR = "./data"


def reformat_data():
    data = read_json(RAW_DATA_PATH)
    reformatted_data = []
    for item in data:
        dataset = item["raw_dataset_name"]
        pipeline = item["pipeline"]
        metafeatures = {
            k: v for k, v in item["metafeatures"].items() if not "time" in k.lower()
        }
        metafeatures["TotalTime"] = item["metafeatures_time"]
        train_time = item["train_time"]
        reformatted_data.append({
            "dataset": dataset,
            "pipeline": pipeline,
            "pipeline_id": item["pipeline_id"],
            "metafeatures": metafeatures,
            "train_accuracy": item["train_accuracy"],
            "test_accuracy": item["test_accuracy"],
            "train_time": train_time,
            "test_time": item["test_time"]
        })
    write_json(reformatted_data, PROCESSED_DATA_PATH, pretty=True)


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


def split_data(data: typing.List[typing.Dict], group_by_key: str, test_size: int, seed: int):
    grouped_data_indices = group_json_objects(data, group_by_key)
    groups = list(grouped_data_indices.keys())

    rnd = random.Random()
    rnd.seed(seed)
    rnd.shuffle(groups)

    train_data = []
    for group in groups[test_size:]:
        for i in grouped_data_indices[group]:
            train_data.append(data[i])

    test_data = []
    for group in groups[:test_size]:
        for i in grouped_data_indices[group]:
            test_data.append(data[i])

    return train_data, test_data


def _extract_tarfile(path):
    assert tarfile.is_tarfile(path)

    dirname = os.path.dirname(path)
    with tarfile.open(path, 'r:*') as tar:
        members = tar.getmembers()
        if len(members) != 1:
            raise ValueError('Expected tar file with 1 member, but got {}'.format(len(members)))
        tar.extractall(os.path.dirname(path))
        extracted_path = os.path.join(dirname, tar.getmembers()[0].name)

    return extracted_path


def get_data(path):
    if tarfile.is_tarfile(path):
        path = _extract_tarfile(path)
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def main():
    extract_data()
    reformat_data()
    drop_nan_metafeatures()
    split_data()


if __name__ == '__main__':
    main()
