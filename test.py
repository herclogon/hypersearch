import torch.nn as nn

from hyperband import Hyperband


def main():

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

    hyper = Hyperband()
    new_model = hyper.tune(model, params)[0]

    print(model)
    print(new_model)


if __name__ == "__main__":
    main()
