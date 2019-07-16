import numpy as np

def rank(values):
    # TODO: ties should be ranked equally
    return list((len(values) - 1 - np.argsort(values)).astype(float))
