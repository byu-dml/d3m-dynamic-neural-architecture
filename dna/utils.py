import typing

import pandas as pd

def rank(values: typing.Sequence) -> typing.Sequence:
    return type(values)((pd.Series(values).rank(ascending=False) - 1))
