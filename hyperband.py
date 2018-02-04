import torch


def Hyperband(object):
    """
    Hyperband is a bandit-based configuration
    evaluation for hyperparameter optimization [1].

    Hyperband is a principled early-stoppping method
    that adaptively allocates resources to randomly
    sampled configurations, quickly eliminating poor
    ones, until a single configuration remains.
    
    References
    ----------
    - [1]: Li et. al., https://arxiv.org/abs/1603.06560
    """
    def __init__(self, model, params, max_iter=81, eta=3):
        """
        Initialize the Hyperband object. Store the model
        along with its hyperparameter configuration space.

        Args
        ----
        - model: the PyTorch model you wish to tune.
        - params: a dictionary where each key is a
          hyperparameter, and each entry is a list of
          the form [s, min, max]. s specifies the scale
          (i.e. linear, log, integer) bounded by the min
          and max.
        - max_iter: maximum amount of iterations that
          you are willing to allocate to a single
          configuration.
        - eta: proportion of configurations discarded
          in each round of Successive Halving.
        """
        self.model = model
        self.params = params
        self.max_iter = max_iter
        self.eta = eta

        # parse the hyperparam config
        params = {
            'conv1_l2': ['log', 5e-5, 5],
            'conv2_l2': ['log', 5e-5, 5],
            'fc1_l2': ['log', 5e-5, 5],
            'fc2_l2': ['log', 5e-5, 5],
        }
