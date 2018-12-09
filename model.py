import torch.nn as nn


def get_base_model():
    layers = []
    layers.append(nn.Linear(784, 100))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(100, 10))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)
    return model
