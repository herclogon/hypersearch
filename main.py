import torch
import torch.nn as nn

from hyperband import Hyperband
from data_loader import get_train_valid_loader


random_seed = 1
use_gpu = False
data_dir = './data/'
batch_size = 64
valid_size = 0.1
shuffle = True
name = 'mnist'


def main():

    # ensure reproducibility
    torch.manual_seed(random_seed)
    kwargs = {}
    if use_gpu:
        torch.cuda.manual_seed(random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    data_loader = get_train_valid_loader(
        data_dir, name, batch_size, random_seed,
        valid_size, shuffle, **kwargs
    )

    # create model
    layers = []
    layers.append(nn.Linear(784, 512))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(512, 256))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 10))
    layers.append(nn.Softmax())
    model = nn.Sequential(*layers)

    params = {
        # '0_dropout': ['uniform', 0.1, 0.5],
        # '0_act': ['choice', ['relu', 'selu', 'elu', 'tanh', 'sigmoid']],
        '0_l2': ['log_uniform', 5e-5, 5],
        # '2_act': ['choice', ['selu', 'elu', 'tanh', 'sigmoid']],
        '4_l1': ['log_uniform', 5e-5, 5],
        'all_act': ['choice', [[0], ['choice', ['selu', 'elu', 'tanh', 'sigmoid']]]],
        'all_dropout': ['choice', [[0], ['uniform', 0.1, 0.5]]],
        'all_batchnorm': ['choice', [0, 1]],
        # 'all_l2': ['log_uniform', 5e-5, 5],
    }

    # instantiate hyperband object
    hyperband = Hyperband(model, params, data_loader)

    # tune
    hyperband.tune()


if __name__ == '__main__':
    main()
