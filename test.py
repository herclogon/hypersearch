import itertools
import numpy as np

def _discretize(r_low, r_high, num_steps=10):
    steps = np.linspace(r_low, r_high, num=num_steps)
    jitter = np.random.uniform((r_high - r_low) / 100., size=steps.shape)
    return steps + jitter

def _get_param_spaces(params):
    layer_hps = []
    for layer_hp, space in params.items():
        layer, hp = layer_hp.rsplit('_', 1)
        if hp == 'l2':
            space = _discretize(space[1], space[2])
        layer_hps.append([(layer, hp, choice) for choice in space])
    return layer_hps

def _yield_hps(params):
    for param_combo in itertools.product(*_get_param_spaces(params)):
        if np.random.random() < 0.8:  # or whatever pruning you wish
            yield param_combo

if __name__ == '__main__':
    params = {
        'conv1_l2': ['log_uniform', 5e-5, 5],
        'fc1_l2': ['log_uniform', 5e-5, 5],
        'all_act': ['choice', ['relu', 'selu', 'elu']]
    }
    for param_combo in _yield_hps(params):
        pass # construct the net (will require if-else)
        print(param_combo)
        break

# sample n sets of hyperparameters for the n models we're gonna train
T = [self.get_random_config() for i in range(n)]

def get_random_config(self):
    """
    Return a set of i.i.d samples from the distributions
    defined over the hyperparameter configuration space.

    This returns a set of hyperparameters for a single
    configuration.

    Returns
    -------
    - hyperparams: a list containing the hyperparameter
      samples sampled from their respective distributions.
    """
    spaces = list(self.params.values())
    hyperparams = [sample_from(space) for space in spaces]
    return hyperparams