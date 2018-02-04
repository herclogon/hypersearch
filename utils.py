import numpy as np

params = {
    'conv1_l2': ['log', 5e-5, 5],
    'conv2_l2': ['log', 5e-5, 5],
    'fc1_l2': ['log', 5e-5, 5],
    'fc2_l2': ['log', 5e-5, 5],
}


def parse_params(params):
    """
    Parse the dictionary of hyperparameters
    and return a new dictionary where the i'th
    key corresponds to a hyperparameter, and
    the i'th value corresponds to 

    """