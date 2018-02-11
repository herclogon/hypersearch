import numpy as np

from utils import sample_from, str2act, find_key

import torch
import torch.nn as nn


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
        - model: the `Sequential()` model you wish to tune.
        - params: a dictionary where each key is a
          hyperparameter, and each entry is a list of
          the form [s, min, max]. s specifies the scale
          (i.e. linear, log, integer) bounded by the min
          and max.
        - data_loaders: [...]

        Returns
        -------
        - results: [...]
        """
        self.model = model
        self.data_loaders = data_loaders
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
                val_losses = [self.run_config(t, r_i) for t in T]

                # keep the best n_i / eta
                T = [
                    T[i] for i in np.argsort(val_losses)[0:int(n_i / self.eta)]
                ]

    def get_random_config(self):
        """
        Build a modified version of the user's model that
        incorporates the new hyperparameters settings defined
        by `hyperparams`.
        """
        layers = []
        used_acts = []
        all_act = False
        all_drop = False
        all_batchnorm = False
        num_layers = len(self.model)

        i = 0
        used_acts.append(self.model[1].__str__())
        for layer_hp in self.params.keys():
            layer, hp = layer_hp.split('_', 1)
            if layer.isdigit():
                layer_num = int(layer)
                diff = layer_num - i
                if diff > 0:
                    for j in range(diff + 1):
                        layers.append(self.model[i+j])
                    i += diff
                    if hp == 'act':
                        space = find_key(
                            self.params, '{}_act'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_act = str2act(hyperp)
                        used_acts.append(new_act.__str__())
                        layers.append(new_act)
                        i += 1
                    elif hp == 'dropout':
                        layers.append(self.model[i])
                        space = find_key(
                            self.params, '{}_drop'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        layers.append(nn.Dropout(p=hyperp))
                    else:
                        pass
                elif diff == 0:
                    layers.append(self.model[i])
                    if hp == 'act':
                        space = find_key(
                            self.params, '{}_act'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_act = str2act(hyperp)
                        used_acts.append(new_act.__str__())
                        layers.append(new_act)
                        i += 1
                    elif hp == 'dropout':
                        i += 1
                        layers.append(self.model[i])
                        space = find_key(
                            self.params, '{}_drop'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        layers.append(nn.Dropout(p=hyperp))
                    else:
                        pass
                else:
                    if hp == 'act':
                        space = find_key(
                            self.params, '{}_act'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_act = str2act(hyperp)
                        used_acts.append(new_act.__str__())
                        layers[i] = new_act
                    elif hp == 'dropout':
                        space = find_key(
                            self.params, '{}_drop'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        layers.append(nn.Dropout(p=hyperp))
                        layers.append(self.model[i])
                    else:
                        pass
                i += 1
            else:
                if (i < num_layers) and (len(layers) < num_layers):
                    for j in range(num_layers-i):
                        layers.append(self.model[i+j])
                    i += 1
                if layer == "all":
                    if hp == "act":
                        space = self.params['all_act']
                        hyperp = sample_from(space)
                        all_act = False if hyperp == [0] else True
                    elif hp == "dropout":
                        space = self.params['all_dropout']
                        hyperp = sample_from(space)
                        all_drop = False if hyperp == [0] else True
                    elif hp == "batchnorm":
                        space = self.params['all_batchnorm']
                        hyperp = sample_from(space)
                        all_batchnorm = True if hyperp == 1 else False
                    else:
                        pass

        used_acts = sorted(set(used_acts), key=used_acts.index)

        if all_act:
            old_act = used_acts[0]
            space = self.params['all_act'][1][1]
            hyperp = sample_from(space)
            new_act = str2act(hyperp)
            used_acts.append(new_act.__str__())
            for i, l in enumerate(layers):
                if l.__str__() == old_act:
                    layers[i] = new_act
        if all_batchnorm:
            self.all_batchnorm = True
            target_acts = used_acts if not all_act else used_acts[1:]
            for i, l in enumerate(layers):
                if l.__str__() in target_acts:
                    if 'Linear' in layers[i-1].__str__():
                        bn = nn.BatchNorm2d(layers[i-1].out_features)
                    else:
                        bn = nn.BatchNorm2d(layers[i-1].out_channels)
                    layers.insert(i+1, bn)
            if 'Linear' in layers[-2].__str__():
                bn = nn.BatchNorm2d(layers[i-1].out_features)
            else:
                bn = nn.BatchNorm2d(layers[i-1].out_channels)
            layers.insert(-1, bn)
        if all_drop:
            self.all_drop = True
            target_acts = used_acts if not all_act else used_acts[1:]
            space = self.params['all_dropout'][1][1]
            hyperp = sample_from(space)
            for i, l in enumerate(layers):
                if l.__str__() in target_acts:
                    layers.insert(i + 1 + all_batchnorm, nn.Dropout(p=hyperp))

        return nn.Sequential(*layers)

    def add_regularization(self, model):
        """
        Setup regularization on model layers based
        on parameter dictionary.
        """
        reg_layers = []
        for k in self.params.keys():
            if k in ["all_l2", "all_l1"]:
                reg_layers.append('all')
            elif k.split('_', 1)[1] in ["l2", "l1"]:
                layer_num = int(k.split('_', 1)[0])
                if layer_num != 0:
                    layer_num += (layer_num // 2) * (
                        self.all_batchnorm + self.all_drop
                    )
                layer_num = str(layer_num)
                l2_reg = True
                if k.split('_', 1)[1] == "l1":
                    l2_reg = False
                space = self.params[k]
                hyperp = sample_from(space)
                reg_layers.append((layer_num, hyperp, l2_reg))
            else:
                pass
        return reg_layers

    def run_config(self, model, num_iters):
        """
        Train a particular hyperparameter configuration for a
        given number of iterations and evaluate the loss on the
        validation set.

        For hyperparameters that have previously been evaluated,
        resume from a previous checkpoint.

        Args
        ----
        - model: 
        - num_iters: an int indicating the number of iterations
          to train the model for.

        Returns
        -------
        - val_losses: [...]
        """
        reg_layers = self.add_regularization(model)

        

