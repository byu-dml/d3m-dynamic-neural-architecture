import json
import tarfile
import os

import numpy as np
import pandas as pd


def _get_standardized_metafeatures(data):
    metafeatures_df = pd.DataFrame([pipe["metafeatures"] for pipe in data])
    # drop individual metafeature times
    metafeatures_df.drop(
        labels=[col for col in metafeatures_df if "time" in col.lower()],
        axis=1,
        inplace = True
    )
    metafeatures_df.replace(
        to_replace = [np.inf, - np.inf], value = np.nan, inplace = True
    )
    metafeatures_df.dropna(axis=1, how="any", inplace=True)

    return metafeatures_df.values.tolist()


def _preprocess_data_for_regression(data):
    metafeatures = _get_standardized_metafeatures(data)
    processed_data = []
    for item, mfs in zip(data, metafeatures):
        # separate dataset from primitives
        dataset, pipeline = item["job_str"].split("___", 1)
        train_time = item["train_fit_time"] + item["train_predict_time"]
        processed_data.append({
            "dataset": dataset,
            "pipeline": pipeline,
            "metafeatures": mfs,
            "train_accuracy": item["train_accuracy"],
            "test_accuracy": item["test_accuracy"],
            "train_time": train_time,
            "test_time": item["test_predict_time"]
        })
    return processed_data


def main():
    data_dir = "./data"
    tar_data_path = f"{data_dir}/complete_pipelines_and_metafeatures.tar.xz"
    regression_data_path = f"{data_dir}/regression_data.json"

    with tarfile.open(tar_data_path, "r:xz") as tar:
        data_path = data_dir + "/" + tar.getnames()[0]
        tar.extractall(data_dir) # todo load directly into memory

    data = json.load(open(data_path, "r"))
    data = _preprocess_data_for_regression(data)
    json.dump(data, open(regression_data_path, "w"), indent=4)
    os.remove(data_path)


if __name__ == '__main__':
    main()
