import typing

import git
import pandas as pd
import torch

def rank(values: typing.Sequence) -> typing.Sequence:
    return type(values)((pd.Series(values).rank(ascending=False) - 1))

def get_git_commit_hash():
    try:
        return git.Repo(search_parent_directories=True).head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        pass

def get_reduction_function(reduction: str):
    if reduction == 'mean':
        return torch.mean
    elif reduction == 'sum':
        return torch.sum
    elif reduction == 'mul':
        return torch.mul
    else:
        raise Exception('No valid reduction was provided\n'
                        'Got \"' + reduction + '\"')
