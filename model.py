import torch.nn as nn

from utils import Reshape


def get_base_model():

    layers = []
    layers.append(nn.Linear(784, 512))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(512, 256))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 128))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(128, 10))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    return model
