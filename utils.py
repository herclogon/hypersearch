import os
import time
import json
import numpy as np

import torch.nn as nn

from numpy.random import uniform, normal, randint, choice


def save_results(r):
    date = time.strftime("%Y%m%d_%H%M%S")
    filename = date + '_results.json'
    param_path = os.path.join('./results/', filename)
    with open(param_path, 'w') as fp:
        json.dump(r, fp, indent=4, sort_keys=True)


def find_key(params, partial_key):
    return next(v for k, v in params.items() if partial_key in k)


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

    np.random.seed(int(time.time() + np.random.randint(0, 300)))

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
        samp = round((samp / space[3]) * space[3])
    return samp


def str2act(a):
    if a == 'relu':
        return nn.ReLU()
    elif a == 'selu':
        return nn.SELU()
    elif a == 'elu':
        return nn.ELU()
    elif a == 'tanh':
        return nn.Tanh()
    elif a == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('[!] Unsupported activation.')


def prepare_dirs(dirs):
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape, -1)


class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
