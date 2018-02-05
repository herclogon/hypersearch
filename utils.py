import numpy as np

from numpy.random import uniform, normal, randint, choice


def sample_from(space):
    """
    Sample a hyperparameter value from a distribution
    defined and parametrized in the list `space`.
    """
    distrs = {
        'choice': choice,
        'randint': randint,
        'uniform': uniform,
        'normal': normal,
    }
    s = space[0]

    log = s.startswith('log_')
    s = s[len('log_'):] if log else s

    quantized = s.startswith('q')
    s = s[1:] if quantized else s

    distr = distrs[s]
    if s == 'choice':
        return distr(space[1])
    samp = distr(space[1], space[2])
    if log:
        samp = np.exp(samp)
    if quantized:
        samp = (samp / space[3]) * space[3]
    return samp
