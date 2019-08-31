import typing

import git
import pandas as pd
import argparse


def rank(values: typing.Sequence) -> typing.Sequence:
    return type(values)((pd.Series(values).rank(ascending=False) - 1))

def get_git_commit_hash():
    try:
        return git.Repo(search_parent_directories=True).head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        pass

def dict_to_namespace(dictionary: dict):
    namespace = argparse.Namespace()
    for k, v in dictionary.items():
        namespace.__setattr__(k, v)
    return namespace
