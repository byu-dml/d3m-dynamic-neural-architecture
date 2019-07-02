import numpy as np

def rank(values):
    return len(values) - 1 - np.argsort(values)
