import numpy as np

from utils import sample_from, str2act, find_key

import torch
import torch.nn as nn

from torch.autograd import Variable


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
    def __init__(self,
                 model,
                 params,
                 data_loader,
                 use_gpu,
                 max_iter=81,
                 eta=4,
                 epoch_scale=True):
        """
        Initialize the Hyperband object.

        Args
        ----
        - model: the `Sequential()` model you wish to tune.
        - params: a dictionary where the key is the hyperparameter
          to tune, and the value is the space from which to randomly
          sample it.
        - data_loader: a tuple containing train and valid iterators
          over the desired dataset.
        - max_iter: maximum amount of iterations that
          you are willing to allocate to a single
          configuration.
        - eta: proportion of configurations discarded
          in each round of Successive Halving (SH).
        - epoch_scale: if True, `max_iter` is computed in
          terms of epochs, else in terms of iterations per
          epoch.
        """
        self.epoch_scale = epoch_scale
        self.max_iter = max_iter
        self.eta = eta
        self.s_max = int(np.log(max_iter) / np.log(eta))
        self.B = (self.s_max+1) * max_iter

        self.model = model
        self.data_loader = data_loader

        self._parse_params(params)

    def _parse_params(self, params):
        """
        Split the user-defined params dictionary
        into its different components.
        """
        self.size_params = {}
        self.net_params = {}
        self.optim_params = {}
        self.reg_params = {}

        size_filter = ["hidden"]
        net_filter = ["act", "dropout", "batchnorm"]
        optim_filter = ["lr", "optim", "batchsize", "momentum"]
        reg_filter = ["l2", "l1"]

        for k, v in params.items():
            if any(s in k for s in size_filter):
                self.size_params[k] = v
            elif any(s in k for s in net_filter):
                self.net_params[k] = v
            elif any(s in k for s in optim_filter):
                self.optim_params[k] = v
            elif any(s in k for s in reg_filter):
                self.reg_params[k] = v
            else:
                raise ValueError("[!] key not supported.")

    def tune(self):
        """
        Tune the hyperparameters of the pytorch model
        using Hyperband.
        """
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
        self.all_batchnorm = False
        self.all_drop = False

        layers = []
        used_acts = []
        all_act = False
        all_drop = False
        all_batchnorm = False
        num_layers = len(self.model)

        i = 0
        used_acts.append(self.model[1].__str__())
        for layer_hp in self.net_params.keys():
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
                            self.net_params, '{}_act'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_act = str2act(hyperp)
                        used_acts.append(new_act.__str__())
                        layers.append(new_act)
                        i += 1
                    elif hp == 'dropout':
                        layers.append(self.model[i])
                        space = find_key(
                            self.net_params, '{}_drop'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        layers.append(nn.Dropout(p=hyperp))
                    else:
                        pass
                elif diff == 0:
                    layers.append(self.model[i])
                    if hp == 'act':
                        space = find_key(
                            self.net_params, '{}_act'.format(layer_num)
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
                            self.net_params, '{}_drop'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        layers.append(nn.Dropout(p=hyperp))
                    else:
                        pass
                else:
                    if hp == 'act':
                        space = find_key(
                            self.net_params, '{}_act'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_act = str2act(hyperp)
                        used_acts.append(new_act.__str__())
                        layers[i] = new_act
                    elif hp == 'dropout':
                        space = find_key(
                            self.net_params, '{}_drop'.format(layer_num)
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
                        space = self.net_params['all_act']
                        hyperp = sample_from(space)
                        all_act = False if hyperp == [0] else True
                    elif hp == "dropout":
                        space = self.net_params['all_dropout']
                        hyperp = sample_from(space)
                        all_drop = False if hyperp == [0] else True
                    elif hp == "batchnorm":
                        space = self.net_params['all_batchnorm']
                        hyperp = sample_from(space)
                        all_batchnorm = True if hyperp == 1 else False
                    else:
                        pass

        used_acts = sorted(set(used_acts), key=used_acts.index)

        if all_act:
            old_act = used_acts[0]
            space = self.net_params['all_act'][1][1]
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
            space = self.net_params['all_dropout'][1][1]
            hyperp = sample_from(space)
            for i, l in enumerate(layers):
                if l.__str__() in target_acts:
                    layers.insert(i + 1 + all_batchnorm, nn.Dropout(p=hyperp))

        sizes = {}
        for k, v in self.size_params.items():
            layer_num = int(k.split("_", 1)[0])
            layer_num += (layer_num // 2) * (
                self.all_batchnorm + self.all_drop
            )
            hyperp = sample_from(v)
            sizes[layer_num] = hyperp

        for layer, size in sizes.items():
            in_dim = layers[layer].in_features
            layers[layer] = nn.Linear(in_dim, size)
            if self.all_batchnorm:
                layers[layer + 2] = nn.BatchNorm2d(size)
            next_layer = layer + (
                2 + self.all_batchnorm + self.all_drop
            )
            out_dim = layers[next_layer].out_features
            layers[next_layer] = nn.Linear(size, out_dim)

        return nn.Sequential(*layers)

    def _add_regularization(self, model):
        """
        Setup regularization on model layers based
        on parameter dictionary.
        """
        reg_layers = []
        for k in self.reg_params.keys():
            if k in ["all_l2", "all_l1"]:
                reg_layers.append('all')
            elif k.split('_', 1)[1] in ["l2", "l1"]:
                layer_num = int(k.split('_', 1)[0])
                layer_num += (layer_num // 2) * (
                    self.all_batchnorm + self.all_drop
                )
                l2_reg = True
                if k.split('_', 1)[1] == "l1":
                    l2_reg = False
                space = self.reg_params[k]
                hyperp = sample_from(space)
                reg_layers.append((layer_num, hyperp, l2_reg))
            else:
                pass
        return reg_layers

    def _get_reg_loss(self, model, reg_layers):
        """
        Compute the regularization loss of the model layers
        as defined by reg_layers.
        """
        reg_loss = Variable(torch.FloatTensor(1), requires_grad=True)
        for layer_num, scale, l2 in reg_layers:
            l1_loss = Variable(torch.FloatTensor(1), requires_grad=True)
            l2_loss = Variable(torch.FloatTensor(1), requires_grad=True)
            if l2:
                for W in model[layer_num].parameters():
                    l2_loss = l2_loss + (W.norm(2) ** 2)
                l2_loss = l2_loss.sqrt()
            else:
                for W in model[layer_num].parameters():
                    l1_loss = l1_loss + W.norm(1)
                l1_loss = l1_loss / 2
            reg_loss = reg_loss + ((l1_loss + l2_loss) * scale)
        return reg_loss

    def run_config(self, model, num_iters):
        """
        Train a particular hyperparameter configuration for a
        given number of iterations and evaluate the loss on the
        validation set.

        For hyperparameters that have previously been evaluated,
        resume from a previous checkpoint.

        Args
        ----
        - model: [...]
        - num_iters: an int indicating the number of iterations
          to train the model for.

        Returns
        -------
        - val_losses: [...]
        """
        # parse reg params
        reg_layers = self._add_regularization(model)

        # get regularization loss
        reg_loss = self._get_reg_loss(model, reg_layers)
