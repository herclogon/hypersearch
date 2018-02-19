import os
import uuid
import numpy as np

from utils import *
from tqdm import tqdm
from data_loader import get_train_valid_loader

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD, Adam
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
    def __init__(self, args, model, params):
        """
        Initialize the Hyperband object.

        Args
        ----
        - args: object containing command line arguments.
        - model: the `Sequential()` model you wish to tune.
        - params: a dictionary where the key is the hyperparameter
          to tune, and the value is the space from which to randomly
          sample it.
        """
        self.args = args
        self.model = model
        self._parse_params(params)

        # hyperband params
        self.epoch_scale = args.epoch_scale
        self.max_iter = args.max_iter
        self.eta = args.eta
        self.s_max = int(np.log(self.max_iter) / np.log(self.eta))
        self.B = (self.s_max+1) * self.max_iter

        print(
            "[*] max_iter: {}, eta: {}, B: {}".format(
                self.max_iter, self.eta, self.B
            )
        )

        # misc params
        self.data_dir = args.data_dir
        self.ckpt_dir = args.ckpt_dir
        self.num_gpu = args.num_gpu
        self.print_freq = args.print_freq

        # data params
        self.data_loader = None
        self.kwargs = {}
        if self.num_gpu > 0:
            self.kwargs = {'num_workers': 1, 'pin_memory': True}
        if 'batch_size' not in self.optim_params:
            self.batch_hyper = False
            self.data_loader = get_train_valid_loader(
                args.data_dir, args.name, args.batch_size,
                args.valid_size, args.shuffle, **self.kwargs
            )

        # optim params
        self.def_optim = args.def_optim
        self.def_lr = args.def_lr
        self.patience = args.patience

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
        optim_filter = ["lr", "optim", "batch_size"]
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
        best_configs = []
        results = {}

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
                n_i = int(n * self.eta ** (-i))
                r_i = int(r * self.eta ** (i))

                tqdm.write(
                    "[*] running {} configs for {} iters each".format(n_i, r_i)
                )

                # run each of the n_i configs for r_i iterations
                val_losses = []
                with tqdm(total=len(T)) as pbar:
                    for t in T:
                        val_loss = self.run_config(t, r_i)
                        val_losses.append(val_loss)
                        pbar.update(1)

                if i == s:
                    sorted_idx = np.argsort(val_losses)
                    T = T[sorted_idx[0]]
                    l = val_losses[sorted_idx[0]]
                    best_configs.append([T, l])

                # remove early stopped configs and keep the best n_i / eta
                T = [
                    T[k] for k in np.argsort(val_losses)
                    if val_losses[k] != 999999
                ]
                T = [
                    T[k] for k in np.argsort(val_losses)[0:int(n_i / self.eta)]
                ]

        best = np.argmax([b[1] for b in best_configs])
        best_model = best_configs[best]
        results["val_loss"] = best_model[1]
        results["params"] = best_model[0].new_params
        results["str"] = best_model[0].__str__()
        return results

    def get_random_config(self):
        """
        Build a mutated version of the user's model that
        incorporates the new hyperparameters settings defined
        by `hyperparams`.
        """
        self.all_batchnorm = False
        self.all_drop = False

        new_params = {}
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
                        new_params["act"] = hyperp
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
                        new_params["drop"] = hyperp
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
                        new_params["act"] = hyperp
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
                        new_params["drop"] = hyperp
                        layers.append(nn.Dropout(p=hyperp))
                    else:
                        pass
                else:
                    if hp == 'act':
                        space = find_key(
                            self.net_params, '{}_act'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_params["act"] = hyperp
                        new_act = str2act(hyperp)
                        used_acts.append(new_act.__str__())
                        layers[i] = new_act
                    elif hp == 'dropout':
                        space = find_key(
                            self.net_params, '{}_drop'.format(layer_num)
                        )
                        hyperp = sample_from(space)
                        new_params["drop"] = hyperp
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
            new_params["all_act"] = hyperp
            new_act = str2act(hyperp)
            used_acts.append(new_act.__str__())
            for i, l in enumerate(layers):
                if l.__str__() == old_act:
                    layers[i] = new_act
        if all_batchnorm:
            self.all_batchnorm = True
            new_params["all_batch"] = True
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
            new_params["all_drop"] = True
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
            new_params["{}_hidden_size".format(layer_num)] = hyperp
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

        mutated = nn.Sequential(*layers)
        self._init_weights_biases(mutated)
        mutated.ckpt_name = str(uuid.uuid4().hex)
        mutated.new_params = new_params
        return mutated

    def _init_weights_biases(self, model):
        """
        If contains mixture of activations, then
        use glorot/xavier initialization. Else:

        - selu: selu init
        - relu: He et. al init
        - other: golorot init
        """
        glorot_all = False
        for key in self.net_params.keys():
            layer, hp = key.split('_', 1)
            if layer.isdigit() and hp == "act":
                glorot_all = True
            elif key == "all_act":
                for i, m in enumerate(model):
                    if not isinstance(m, 
                        (nn.Linear, nn.BatchNorm2d, nn.Dropout)
                    ):
                        if i < len(model) - 1:
                            act = model[i].__str__()
                            act = act.lower().split("(")[0]
                            break
        if glorot_all:
            for m in model:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant(m.weight, 1)
                    nn.init.constant(m.bias, 0)
        else:
            if act == "relu":
                for m in model:
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal(m.weight)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant(m.weight, 1)
                        nn.init.constant(m.bias, 0)
            elif act == "selu":
                for m in model:
                    if isinstance(m, nn.Linear):
                        n = m.out_features
                        nn.init.normal(m.weight, 
                            mean=0, std=np.sqrt(1. / n)
                        )
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant(m.weight, 1)
                        nn.init.constant(m.bias, 0)
            else:
                for m in model:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal(m.weight)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant(m.weight, 1)
                        nn.init.constant(m.bias, 0)

    def _check_bn_drop(self, model):
        names = []
        count = 0
        for layer in model.named_children():
            names.append(layer[1].__str__().split("(")[0])
        names = list(set(names))
        if any("Dropout" in s for s in names):
            count += 1
        if any("BatchNorm" in s for s in names):
            count += 1
        return count

    def _add_reg(self, model):
        """
        Setup regularization on model layers based
        on parameter dictionary.
        """
        offset = self._check_bn_drop(model)
        reg_layers = []
        for k in self.reg_params.keys():
            if k in ["all_l2", "all_l1"]:
                l2_reg = False
                if k == "all_l2":
                    l2_reg = True
                num_lin_layers = int(
                    ((len(self.model) - 2) / 2) + 1
                )
                j = 0
                for i in range(num_lin_layers):
                    space = self.reg_params[k]
                    hyperp = sample_from(space)
                    reg_layers.append((j, hyperp, l2_reg))
                    j += 2 + offset
            elif k.split('_', 1)[1] in ["l2", "l1"]:
                layer_num = int(k.split('_', 1)[0])
                layer_num += (layer_num // 2) * (offset)
                l2_reg = True
                if k.split('_', 1)[1] == "l1":
                    l2_reg = False
                space = self.reg_params[k]
                hyperp = sample_from(space)
                reg_layers.append((layer_num, hyperp, l2_reg))
            else:
                pass
        model.new_params["reg_layers"] = reg_layers
        return reg_layers

    def _get_reg_loss(self, model, reg_layers):
        """
        Compute the regularization loss of the model layers
        as defined by reg_layers.
        """
        dtype = torch.FloatTensor if self.num_gpu == 0 else torch.cuda.FloatTensor
        reg_loss = Variable(torch.zeros(1), requires_grad=True).type(dtype)
        for layer_num, scale, l2 in reg_layers:
            l1_loss = Variable(torch.zeros(1), requires_grad=True).type(dtype)
            l2_loss = Variable(torch.zeros(1), requires_grad=True).type(dtype)
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

    def _get_optimizer(self, model):
        """
        Setup optimizer and learning rate.
        """
        lr = self.def_lr
        name = self.def_optim
        if "optim" in self.optim_params:
            space = self.optim_params['optim']
            name = sample_from(space)
        if "lr" in self.optim_params:
            space = self.optim_params['lr']
            lr = sample_from(space)
        if name == "sgd":
            opt = SGD
        elif name == "adam":
            opt = Adam
        model.new_params["optim"] = name
        model.new_params["lr"] = lr
        optim = opt(model.parameters(), lr=lr)
        return optim

    def run_config(self, model, num_iters):
        """
        Train a particular hyperparameter configuration for a
        given number of iterations and evaluate the loss on the
        validation set.

        For hyperparameters that have previously been evaluated,
        resume from a previous checkpoint.

        Args
        ----
        - model: the mutated model to train.
        - num_iters: an int indicating the number of iterations
          to train the model for.

        Returns
        -------
        - val_loss: the lowest validaton loss achieved.
        """
        # load the most recent checkpoint if it exists
        try:
            ckpt = self._load_checkpoint(model.ckpt_name)
            model.load_state_dict(ckpt['state_dict'])
        except FileNotFoundError:
            pass

        if self.num_gpu > 0:
            model = model.cuda()

        # parse reg params
        reg_layers = self._add_reg(model)

        # setup train loader
        if self.data_loader is None:
            self.batch_hyper = True
            space = self.optim_params['batch_size']
            batch_size = sample_from(space)
            tqdm.write("batch size: {}".format(batch_size))
            self.data_loader = get_train_valid_loader(
                self.data_dir, self.args.name,
                batch_size, self.args.valid_size,
                self.args.shuffle, **self.kwargs
            )

        # training logic
        min_val_loss = 999999
        counter = 0
        num_epochs = int(num_iters) if self.epoch_scale else 1
        num_passes = None if self.epoch_scale else num_iters
        for epoch in range(num_epochs):
            self._train_one_epoch(model, num_passes, reg_layers)
            val_loss = self._validate_one_epoch(model)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                counter = 0
            else:
                counter += 1
            if counter > self.patience:
                return 999999
        if self.batch_hyper:
            self.data_loader = None
        state = {
            'state_dict': model.state_dict(),
            'min_val_loss': min_val_loss,
        }
        self._save_checkpoint(state, model.ckpt_name)
        return min_val_loss

    def _train_one_epoch(self, model, num_passes, reg_layers):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        If `num_passes` is not None, the model is trained for
        `num_passes` mini-batch iterations.
        """
        model.train()

        # setup optimizer
        optim = self._get_optimizer(model)

        train_loader = self.data_loader[0]
        num_train = len(train_loader.sampler.indices)
        for i, (x, y) in enumerate(train_loader):
            if num_passes is not None:
                if i > num_passes:
                    return
            if self.num_gpu > 0:
                x, y = x.cuda(), y.cuda()
            batch_size = x.shape[0]
            x = x.view(batch_size, -1)
            x, y = Variable(x), Variable(y)
            optim.zero_grad()
            output = model(x)
            loss = F.nll_loss(output, y)
            reg_loss = self._get_reg_loss(model, reg_layers)
            comp_loss = loss + reg_loss
            comp_loss.backward()
            optim.step()

    def _validate_one_epoch(self, model):
        """
        Evaluate the model on the validation set.
        """
        model.eval()

        val_loader = self.data_loader[1]
        num_valid = len(val_loader.sampler.indices)
        val_loss = 0.
        for i, (x, y) in enumerate(val_loader):
            if self.num_gpu > 0:
                x, y = x.cuda(), y.cuda()
            x = x.view(x.size(0), -1)
            x, y = Variable(x), Variable(y)
            output = model(x)
            val_loss += F.nll_loss(output, y, size_average=False).data[0]

        val_loss /= num_valid
        return val_loss

    def _save_checkpoint(self, state, name):
        """
        Save a copy of the model.
        """
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

    def _load_checkpoint(self, name):
        """
        Load the latest model checkpoint.
        """
        filename = name + '.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)
        return ckpt
