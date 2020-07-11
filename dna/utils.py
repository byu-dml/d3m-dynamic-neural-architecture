import collections
import typing
import json

import git
import pandas as pd
import numpy as np

from dna import constants


def rank(values: typing.Sequence) -> typing.Sequence:
    return type(values)((pd.Series(values).rank(ascending=False) - 1))


def get_git_commit_hash():
    try:
        return git.Repo(search_parent_directories=True).head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        pass


def flatten(d, parent_key='', sep='.'):
    # TODO: handle iterables
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def inflate(d, sep='.'):
    """Inverse of flatten"""
    result = {}
    for key, value in d.items():
        sub_result = result
        key_parts = key.split(sep)
        for sub_key in key_parts[:-1]:
            if sub_key not in sub_result:
                sub_result[sub_key] = {}
            sub_result = sub_result[sub_key]
        sub_result[key_parts[-1]] = value
    return result


def transpose_jagged_2darray(jagged_2darray: typing.Iterable[typing.Iterable]) -> typing.Dict[int, typing.List]:
    """Transposes a 2D jagged array into a dict mapping column index to a list of row values.

    For example:
        [
            [0, 1],
            [2, 3, 4],
            [5],
            [6, 7, 8, 9],
        ]
    becomes
        {
            0: [0, 2, 5, 6],
            1: [1, 3, 7],
            2: [4, 8],
            3: [9],
        }
    """
    transpose = {}
    for row in jagged_2darray:
        for i, value in enumerate(row):
            if i not in transpose:
                transpose[i] = []
            transpose[i].append(value)
    return transpose


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that handles NumPy Arrays"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def has_path(data, path) -> bool:
    """
    Returns `True` if `data` has the key path identified
    by `path`. The keys can be dictionary keys or list/tuple
    indices.
    """
    walk = data
    for key in path:
        if isinstance(walk, dict):
            if key in walk:
                walk = walk[key]
            else:
                return False
        elif isinstance(walk, list) or isinstance(walk, tuple):
            if isinstance(key, int) and key < len(walk):
                walk = walk[key]
            else:
                return False
        else:
            return False
    return True


def get_primitive_one_hot_mapping(
    data, *, pipeline_key: str, steps_key: str, prim_name_key: str
) -> typing.Tuple[dict, int]:
    primitive_names = set()

    # Get a set of all the primitives in the data set
    for instance in data:
        primitives = instance[pipeline_key][steps_key]
        for primitive in primitives:
            primitive_name = primitive[prim_name_key]
            primitive_names.add(primitive_name)
        
    primitive_names = sorted(primitive_names)
    primitive_names.append(constants.UNKNOWN)  # to handle unseen primitives

    # Get one hot encodings of all the primitives
    n_primitives = len(primitive_names)
    encoding = np.identity(n=n_primitives)

    # Create a mapping of primitive names to one hot encodings
    primitive_name_to_enc = {}
    for (primitive_name, primitive_encoding) in zip(primitive_names, encoding):
        primitive_name_to_enc[primitive_name] = primitive_encoding

    return primitive_name_to_enc, n_primitives
