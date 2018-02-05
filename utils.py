import numpy as np

from numpy import random


def parse_config(val):
    """
    Parse a list containing the distribution
    to sample from with associated parameters.
    """
    s = val[0]

    if s == 'randint':
        return randint(val[1])
    elif s == 'uniform':
        return uniform(val[1], val[2])
    elif s == 'quniform':
        return round(uniform(val[1], val[2]) / val[3]) * val[3]
    elif s == 'log_uniform':
        return np.exp(uniform(val[1], val[2]))
    elif s == 'log_quniform':
        return round(np.exp(uniform(val[1], val[2])) / val[3]) * val[3]
    elif s == 'normal':
        return normal(val[1], val[2])
    elif s == 'qnormal':
        return round(normal(val[1], val[2]) / val[3]) * val[3]
    elif s == 'log_normal':
        return np.exp(normal(val[1], val[2]))
    elif s == 'log_qnormal':
        return round(np.exp(normal(val[1], val[2])) / val[3]) * val[3]
    else:
        raise ValueError('[!] Unsupported distribution')
