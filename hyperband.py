import numpy as np

from utils import parse_config


class Hyperband(object):
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
    def __init__(self, max_iter=81, eta=3):
        """
        Initialize the Hyperband object.

        Args
        ----
        - max_iter: maximum amount of iterations that
          you are willing to allocate to a single
          configuration.
        - eta: proportion of configurations discarded
          in each round of Successive Halving (SH).
        """
        # max number of iterations allocated to a given config
        self.max_iter = max_iter

        # proportion of configs discarded in each round of SH
        self.eta = eta

        # number of unique executions of SH
        self.s_max = int(np.log(max_iter) / np.log(eta))

        # total number of iterations per execution of SH
        self.B = (self.s_max+1) * max_iter

    def tune(self, model, params, data_loaders):
        """
        Tune the hyperparameters of the pytorch model
        using Hyperband.

        Args
        ----
        - model: the PyTorch model you wish to tune.
        - params: a dictionary where each key is a
          hyperparameter, and each entry is a list of
          the form [s, min, max]. s specifies the scale
          (i.e. linear, log, integer) bounded by the min
          and max.

        Returns
        -------
        - results: [...]
        """
        self.model = model
        self.params = params

        # finite horizon outerloop
        for s in reversed(range(self.s_max + 1)):
            # initial number of configs
            n = int(
                np.ceil(
                    self.B / self.max_iter / (s+1) * self.eta ** s
                )
            )
            # initial number of iterations to run the n configs for
            r = self.max_iter * self.eta ** (-s)

            # finite horizon SH with (n, r)
            T = [self.get_random_config() for i in range(n)]
            for i in range(s + 1):
                n_i = n * self.eta ** (-i)
                r_i = r * self.eta ** (i)

                # run each of the n_i configs for r_i iterations
                val_losses = [self.run_config(r_i, t) for t in T]

                # keep the best n_i / eta
                T = [
                    T[i] for i in np.argsort(val_losses)[0:int(n_i / self.eta)]
                ]

    def get_random_config(self):
        """
        Return a set of i.i.d samples from the distributions
        defined over the hyperparameter configuration space.

        Returns
        -------
        - hyperparams: a list of size n containing the hyperparams
          samples from their respective distributions.
        """
        vals = list(self.params.values())
        hyperparams = [parse_config(v) for v in vals]
        return hyperparams

    def run_config(self, num_iters, hyperparams):
        """
        Train a particular hyperparameter configuration for a
        given number of iterations and evaluate the loss on the
        validation set.

        For hyperparameters that have previously been evaluated,
        resume from a previous checkpoint.

        Args
        ----
        - num_iters: an int indicating the number of iterations
          to train the model for.
        - hyperparams: hyperparameters

        Returns
        -------
        - val_losses:
        """
        pass
