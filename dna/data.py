import json
import os
import random
import tarfile
import typing
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from dna.utils import get_values_by_path


def group_json_objects(json_objects: typing.List[typing.Dict], group_key: str) -> dict:
    """
    Groups JSON data by group_key.

    Parameters:
    -----------
    json_objects: List[Dict], JSON compatible list of objects.
    group_key: str, json_objects is grouped by group_key. group_key must be a
        key into each object in json_objects and the corresponding value must
        be hashable. group_key can be a '.' delimited string to access deeply
        nested fields.

    Returns:
    --------
    A dict with key being a group and the value is a list of indices into
    json_objects.
    """
    grouped_objects = {}
    for i, obj in enumerate(json_objects):
        group = obj
        for key_part in group_key.split('.'):
            group = group[key_part]
        if not group in grouped_objects:
            grouped_objects[group] = []
        grouped_objects[group].append(i)
    return grouped_objects


def group_data_using_grouped_indices(
    data: typing.List[typing.Dict], grouped_data_indices: dict, groups: list
) -> dict:
    """
    Takes `data`, `grouped_data_indices` (a mapping of group names to data indices),
    and `groups` (the names of the groups of data we want to keep), and returns a
    mapping of group names to the actual data instances (not just the indices), to
    make the data more easily useable.
    """
    grouped_data = defaultdict(list)
    for group in groups:
        for i in grouped_data_indices[group]:
            grouped_data[group].append(data[i])
    return grouped_data


def flatten_grouped_data(
    grouped_data: dict,
) -> typing.List[typing.Dict]:
    """
    Flattens a mapping of group names to group members to just
    a list of members.
    """
    # Concatenate all the group lists into one list.
    return list(itertools.chain.from_iterable(grouped_data.values()))


def get_coverage(
    data: list,
    coverage_key: str
) -> set:
    """
    Gets a set of all unique values found under `coverage_key`
    for all the data within `data`.
    """
    coverage_path = coverage_key.split(".")
    return set(get_values_by_path(data, coverage_path))


def ensure_coverage(
    train_data_grouped: dict,
    test_data_grouped: dict,
    coverage_key: str,
    seed: int
) -> tuple:
    """
    Takes the group split `split_data_by_group` has found and ensures that
    all unique values of `coverage_key` are found at least once in the training
    set. Mutates the train and test sets to ensure coverage.
    """
    rng = random.Random()
    rng.seed(seed)

    train_coverage = get_coverage(flatten_grouped_data(train_data_grouped), coverage_key)
    test_coverage = get_coverage(flatten_grouped_data(test_data_grouped), coverage_key)

    while len(test_coverage - train_coverage) > 0:
        # The test set has unique values for the coverage key that are not
        # found inside the training set.

        # 1. Find groups in the test set that have primitives not used in
        #    the training set.
        test_data_grouped_not_covered = {
            group_name: instances
            for group_name, instances
            in test_data_grouped.items()
            if len(get_coverage(instances, coverage_key) - train_coverage) > 0
        }

        # 2. Randomly select one of those groups, and randomly select a training
        #    set group, and swap them.
        test_group_name, test_group_instances = rng.choice(list(test_data_grouped_not_covered.items()))
        train_group_name, train_group_instances = rng.choice(list(train_data_grouped.items()))
        del train_data_grouped[train_group_name]
        train_data_grouped[test_group_name] = test_group_instances
        del test_data_grouped[test_group_name]
        test_data_grouped[train_group_name] = train_group_instances

        # 3. Repeat that process until there are no primitives in the test set
        #    that are not also present in the training set.
        train_coverage = get_coverage(flatten_grouped_data(train_data_grouped), coverage_key)
        test_coverage = get_coverage(flatten_grouped_data(test_data_grouped), coverage_key)

    return train_data_grouped, test_data_grouped    


def split_data_by_group(
    data: typing.List[typing.Dict],
    group_by_key: str,
    coverage_key: str,
    test_size: typing.Union[int, float],
    seed: int,
):
    grouped_data_indices = group_json_objects(data, group_by_key)
    groups = list(grouped_data_indices.keys())

    if 0 < test_size < 1:
        test_size = int(round(test_size * len(groups)))
    if test_size <= 0 or len(groups) <= test_size:
        raise ValueError('invalid test size: {}'.format(test_size))

    rng = random.Random()
    rng.seed(seed)
    rng.shuffle(groups)

    train_groups = groups[test_size:]
    assert len(train_groups) == len(groups) - test_size
    train_data_grouped = group_data_using_grouped_indices(data, grouped_data_indices, train_groups)

    test_groups = groups[:test_size]
    assert len(test_groups) == test_size
    test_data_grouped = group_data_using_grouped_indices(data, grouped_data_indices, test_groups)

    train_data_grouped, test_data_grouped = ensure_coverage(
        train_data_grouped, test_data_grouped, coverage_key, seed
    )

    train_data = flatten_grouped_data(train_data_grouped)
    test_data = flatten_grouped_data(test_data_grouped)

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


class DropMissingValues:

    def __init__(self, values_to_drop=[]):
        self.values_to_drop = values_to_drop

    def fit(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        for key, is_missing in pd.DataFrame(data).isna().any().iteritems():
            if is_missing:
                self.values_to_drop.append(key)

    def predict(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        for instance in data:
            for key in self.values_to_drop:
                instance.pop(key, None)
        return data


class StandardScaler:
    """
    Transforms data by subtracting the mean and scaling by the standard
    deviation. Drops columns that have 0 standard deviation. Clips values to
    numpy resolution, min, and max.
    """

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        values_map = {}
        for instance in data:
            for key, value in instance.items():
                if key not in values_map:
                    values_map[key] = []
                values_map[key].append(value)

        self.means = {}
        self.stds = {}
        for key, values in values_map.items():
            self.means[key] = np.mean(values)
            self.stds[key] = np.std(values, ddof=1)

    def predict(
        self, data: typing.List[typing.Dict[str, typing.Union[int, float]]]
    ):
        if self.means is None or self.stds is None:
            raise Exception('StandardScaler not fit')

        transformed_data = []
        for instance in data:
            transformed_instance = {}
            for key, value in instance.items():
                if self.stds[key] != 0:  # drop columns with 0 std dev
                    transformed_instance[key] = (value - self.means[key]) / self.stds[key]

            transformed_data.append(transformed_instance)

        return transformed_data


def encode_dag(dag: typing.Sequence[typing.Sequence[typing.Any]]):
    """
    Converts a directed acyclic graph DAG) to a string. If two DAGs have the same encoding string, then they are equal.
    However, two isomorphic DAGs may have different encoding strings.

    Parameters
    ----------
    dag: typing.List[typing.List[typing.Any]]
        A representation of a dag. Each element in the outer list represents a vertex. Each inner list or vertex
        contains a reference to the outer list, representing edges.
    """
    return ''.join(''.join(str(edge) for edge in vertex) for vertex in dag)


def filter_metafeatures(metafeatures: dict, metafeature_subset: str):
    landmarker_key_part1 = 'ErrRate'
    landmarker_key_part2 = 'Kappa'

    metafeature_keys = list(metafeatures.keys())

    if metafeature_subset == 'landmarkers':
        for metafeature_key in metafeature_keys:
            if landmarker_key_part1 not in metafeature_key and landmarker_key_part2 not in metafeature_key:
                metafeatures.pop(metafeature_key)
    elif metafeature_subset == 'non-landmarkers':
        for metafeature_key in metafeature_keys:
            if landmarker_key_part1 in metafeature_key or landmarker_key_part2 in metafeature_key:
                metafeatures.pop(metafeature_key)

    return metafeatures


def preprocess_data(train_data, test_data, metafeature_subset: str):
    for instance in train_data:
        instance['pipeline_id'] = instance['pipeline']['id']

    for instance in test_data:
        instance['pipeline_id'] = instance['pipeline']['id']

    train_metafeatures = []
    for instance in train_data:
        metafeatures = filter_metafeatures(instance['metafeatures'], metafeature_subset)
        train_metafeatures.append(metafeatures)
        for step in instance['pipeline']['steps']:
            step['name'] = step['name'].replace('.', '_')

    test_metafeatures = []
    for instance in test_data:
        metafeatures = filter_metafeatures(instance['metafeatures'], metafeature_subset)
        test_metafeatures.append(metafeatures)
        for step in instance['pipeline']['steps']:
            step['name'] = step['name'].replace('.', '_')

    # drop metafeature if missing for any instance
    dropper = DropMissingValues()
    dropper.fit(train_metafeatures)
    train_metafeatures = dropper.predict(train_metafeatures)
    test_metafeatures = dropper.predict(test_metafeatures)

    # scale data to unit mean and unit standard deviation
    scaler = StandardScaler()
    scaler.fit(train_metafeatures)
    train_metafeatures = scaler.predict(train_metafeatures)
    test_metafeatures = scaler.predict(test_metafeatures)

    # convert from dict to list
    for instance, mf_instance in zip(train_data, train_metafeatures):
        instance['metafeatures'] = [value for key, value in sorted(mf_instance.items())]
        pipeline_dag = (step['inputs'] for step in instance['pipeline']['steps'])
        instance['pipeline_structure'] = encode_dag(pipeline_dag)

    for instance, mf_instance in zip(test_data, test_metafeatures):
        instance['metafeatures'] = [value for key, value in sorted(mf_instance.items())]
        pipeline_dag = (step['inputs'] for step in instance['pipeline']['steps'])
        instance['pipeline_structure'] = encode_dag(pipeline_dag)

    return train_data, test_data


class Dataset(torch.utils.data.Dataset):
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
        self, data: typing.List[typing.Dict], features_key: str,
        target_key: str, y_dtype: typing.Any, device: str
    ):
        self.data = data
        self.features_key = features_key
        self.target_key = target_key
        self.y_dtype = y_dtype
        self.device = device

    def __getitem__(self, item: int):
        x = torch.tensor(self.data[item][self.features_key], dtype=torch.float32, device=self.device)
        y = torch.tensor(self.data[item][self.target_key], dtype=self.y_dtype, device=self.device)
        return x, y

    def __len__(self):
        return len(self.data)


class RandomSampler(torch.utils.data.Sampler):
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
        self, data: typing.List[typing.Dict], group_key: str,
        dataset_class: typing.Type[torch.utils.data.Dataset], dataset_params: dict,
        batch_size: int, drop_last: bool, shuffle: bool, seed: int
    ):
        self.data = data
        self.group_key = group_key
        self.dataset_class = dataset_class
        self.dataset_params = dataset_params
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        self._random = random.Random()
        self._random.seed(seed)
        self.old_indices = []

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
            self.old_indices += group_indices
            group_data = [self.data[i] for i in group_indices]
            group_dataset = self.dataset_class(group_data, **self.dataset_params)
            new_dataloader = self._get_data_loader(
                group_dataset
            )
            self._group_dataloaders[group] = new_dataloader

    def _get_data_loader(self, data):
        if self.shuffle:
            sampler = RandomSampler(len(data), self._randint())
        else:
            sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset = data,
            sampler =  sampler,
            batch_size = self.batch_size,
            drop_last = self.drop_last
        )
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
            self._group_batches += [group] * len(group_dataloader)
        if self.shuffle:
            self._random.shuffle(self._group_batches)

    def get_group_ordering(self):
        """
        Returns the indices needed to invert the ordering on the input data generated by the grouping mechanism. This
        method does not work if shuffle or drop last has been set to true.
        """
        if self.shuffle or self.drop_last:
            raise NotImplementedError('cannot ungroup data when shuffle is true or drop_last is true')
        return np.argsort(np.array(self.old_indices))

    def __iter__(self):
        return iter(self._iter())

    def _iter(self):
        group_dataloader_iters = {}
        for group in self._group_batches:
            if not group in group_dataloader_iters:
                group_dataloader_iters[group] = iter(
                    self._group_dataloaders[group]
                )
            x_batch, y_batch = next(group_dataloader_iters[group])
            # since all pipeline are the same in this group, just grab one of them
            pipeline = self._group_dataloaders[group].dataset.data[0]["pipeline"]
            yield (group, pipeline, x_batch), y_batch
        raise StopIteration()

    def __len__(self):
        return len(self._group_batches)


class PMFDataLoader(object):

    def __init__(
        self, data, n_x, n_y, pipeline_encoder, dataset_encoder, pipeline_id_mapper, dataset_id_mapper, device="cuda:0"
    ):
        # assign functions for mapping
        self.pipeline_id_mapper = pipeline_id_mapper
        self.dataset_id_mapper = dataset_id_mapper
        self.encode_pipeline = pipeline_encoder
        self.dataset_encoder = dataset_encoder
         # encode the pipeline dataset mapping
        x_data = self.encode_pipeline_dataset(data)
        y_data = [instance["test_f1_macro"] for instance in data]
        # Build the matrix using the x and y data
        self.matrix = torch.zeros([n_x, n_y], device=device)
        for index, value in enumerate(y_data):
            self.matrix[x_data[index]["pipeline_id_embedding"]][x_data[index]["dataset_id_embedding"]] = value
        self.used = False
        self.n = len(y_data)

    def __len__(self):
        return 1

    def __iter__(self):
        # only return one object: the matrix
        if not self.used:
            yield(None, self.matrix)
        raise StopIteration()

    def encode_pipeline_dataset(self, data):
        """
        Creates the embeddings for the dataset
        """
        try:
            x_data = []
            for instance in data:
                x_data.append({"pipeline_id_embedding": self.encode_pipeline(instance["pipeline"]["id"]),
                            "dataset_id_embedding": self.dataset_id_mapper[instance["dataset_id"]]})
            return x_data

        except KeyError as e:
            raise KeyError("Pipeline/Dataset ID was not in the mapper. Perhaps the pipeline/dataset id was not in the training set? Error: {}".format(e))

    def get_predictions_from_matrix(self, x_data, matrix):
        predictions = []
        for index, item in enumerate(x_data):
            predict_value = matrix[self.encode_pipeline(item["pipeline_id"])][self.dataset_id_mapper[item["dataset_id"]]].item()
            predictions.append(predict_value)

        return predictions


class PMFDataset(Dataset):
    # needed to encode the pipelines and datasets for the embeded layers.  Used with GroupDataLoader.
    def  __init__(
        self, data: typing.List[typing.Dict], features_key: str,
        target_key: str, y_dtype: typing.Any, device: str, encoding_function
    ):
        super().__init__(
            data, features_key, target_key, y_dtype, device
        )
        self.dataset_encoding_function = encoding_function

    def __getitem__(self, item: int):
        x = self.dataset_encoding_function(self.data[item][self.features_key]).to(self.device)
        y = torch.tensor(self.data[item][self.target_key], dtype=self.y_dtype, device=self.device)
        return x, y


class RNNDataset(Dataset):

    def __init__(self, data: dict, features_key: str, target_key: str, y_dtype, device: str):
        super().__init__(data, features_key, target_key, y_dtype, device)
        self.pipeline_key = 'pipeline'
        self.steps_key = 'steps'

    def __getitem__(self, index):
        (x, y) = super().__getitem__(index)
        item = self.data[index]
        encoded_pipeline = torch.tensor(
            item[self.pipeline_key][self.steps_key], dtype=torch.float32, device=self.device
        )
        return (encoded_pipeline, x, y)


class RNNDataLoader(GroupDataLoader):

    def __init__(
        self, data: dict, group_key: str, dataset_params: dict, batch_size: int, drop_last: bool, shuffle: bool,
        seed: int, primitive_to_enc: dict, pipeline_key: str, steps_key: str, prim_name_key: str,
        pipeline_structures: dict = None
    ):
        super().__init__(data, group_key, RNNDataset, dataset_params, batch_size, drop_last, shuffle, seed)
        self.pipeline_structures = pipeline_structures

        if not self._pipelines_encoded(data, pipeline_key, steps_key):
            self._encode_pipelines(data, primitive_to_enc, pipeline_key, steps_key, prim_name_key)

    def _encode_pipelines(self, data, primitive_name_to_enc, pipeline_key, steps_key, prim_name_key):
        for instance in data:
            pipeline = instance[pipeline_key][steps_key]
            encoded_pipeline = self._encode_pipeline(pipeline, primitive_name_to_enc, prim_name_key)
            instance[pipeline_key][steps_key] = encoded_pipeline

    @staticmethod
    def _encode_pipeline(pipeline, primitive_to_enc, prim_name_key):
        # Create a tensor of encoded primitives
        encoding = []
        for primitive in pipeline:
            primitive_name = primitive[prim_name_key]
            try:
                encoded_primitive = primitive_to_enc[primitive_name]
            except():
                raise KeyError('A primitive in this data set is not in the primitive encoding')

            encoding.append(encoded_primitive)
        return encoding

    @staticmethod
    def _pipelines_encoded(data, pipeline_key, steps_key):
        primitive_in_pipeline = data[0][pipeline_key][steps_key][0]
        return type(primitive_in_pipeline) == np.ndarray

    def _iter(self):
        group_dataloader_iters = {}
        for group in self._group_batches:
            if not group in group_dataloader_iters:
                group_dataloader_iters[group] = iter(self._group_dataloaders[group])

            # Get a batch of encoded pipelines, metafeatures, and targets
            (pipeline_batch, x_batch, y_batch) = next(group_dataloader_iters[group])

            if self.pipeline_structures is not None:
                # Get the structure of the pipelines in this group so the RNN can parse the pipeline
                group_structure = self.pipeline_structures[group]

                yield ((group_structure, pipeline_batch, x_batch), y_batch)
            else:
                # Don't return a pipeline structure and the RNN will have to treat it like a straight pipeline
                yield((pipeline_batch, x_batch), y_batch)
        raise StopIteration()
