from config import get_config
from utils import prepare_dirs
from hyperband import Hyperband
from model import get_base_model


def main(config):

    # ensure directories are setup
    dirs = [config.data_dir, config.ckpt_dir]
    prepare_dirs(dirs)

    # create base model
    model = get_base_model()

    print(model)

    # define params
    params = {
        # '0_dropout': ['uniform', 0.1, 0.5],
        # '0_act': ['choice', ['relu', 'selu', 'elu', 'tanh', 'sigmoid']],
        # '0_l2': ['log_uniform', 1e-1, 2],
        # '2_act': ['choice', ['selu', 'elu', 'tanh', 'sigmoid']],
        '2_l1': ['log_uniform', 1e-1, 2],
        '2_hidden': ['quniform', 512, 1000, 1],
        '4_hidden': ['quniform', 128, 512, 1],
        'all_act': ['choice', [[0], ['choice', ['selu', 'elu', 'tanh']]]],
        'all_dropout': ['choice', [[0], ['uniform', 0.1, 0.5]]],
        'all_batchnorm': ['choice', [0, 1]],
        'all_l2': ['log_uniform', 5e-5, 5],
        # 'optim': ['choice', ["adam", "sgd"]],
        # 'lr': ['choice', [1e-3, 1e-4]],
    }

    # instantiate hyperband object
    hyperband = Hyperband(config, model, params)

    # tune
    hyperband.tune()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
