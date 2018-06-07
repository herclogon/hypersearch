import config

from hyperband import Hyperband
from model import get_base_model
from utils import prepare_dirs, save_results


def main(args):

    # ensure directories are setup
    dirs = [args.data_dir, args.ckpt_dir]
    prepare_dirs(dirs)

    # create base model
    model = get_base_model()

    # define params
    params = {
        # '0_dropout': ['uniform', 0.1, 0.5],
        # '0_act': ['choice', ['relu', 'selu', 'elu', 'tanh', 'sigmoid']],
        # '0_l2': ['log_uniform', 1e-1, 2],
        # '2_act': ['choice', ['selu', 'elu', 'tanh', 'sigmoid']],
        # '2_l1': ['log_uniform', 1e-1, 2],
        # '2_hidden': ['quniform', 512, 1000, 1],
        # '4_hidden': ['quniform', 128, 512, 1],
        # 'all_act': ['choice', [[0], ['choice', ['selu', 'elu', 'tanh']]]],
        'all_dropout': ['choice', [[0], ['uniform', 0.1, 0.5]]],
        # 'all_batchnorm': ['choice', [0, 1]],
        'all_l2': ['uniform', 1e-8, 1e-5],
        # 'optim': ['choice', ["adam", "sgd"]],
        # 'lr': ['uniform', 1e-3, 8e-3],
        # 'batch_size': ['quniform', 32, 128, 1]
    }

    # instantiate hyperband object
    hyperband = Hyperband(args, model, params)

    # tune
    results = hyperband.tune()

    # dump results
    save_results(results)


if __name__ == '__main__':
    args, unparsed = config.get_args()
    main(args)
