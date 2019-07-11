import numpy as np

def rank(values):
    return list((len(values) - 1 - np.argsort(values)).astype(float))
