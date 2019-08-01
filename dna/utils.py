import typing

import git
import pandas as pd


def rank(values: typing.Sequence) -> typing.Sequence:
    return type(values)((pd.Series(values).rank(ascending=False) - 1))

def get_git_commit_hash():
    try:
        return git.Repo(search_parent_directories=True).head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        pass
