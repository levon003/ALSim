import scipy
import numpy as np


def compute_ci(n, std, cl=0.95):
    z = scipy.stats.norm.ppf(1 - (1 - cl) / 2)
    return z * std / np.sqrt(n)
