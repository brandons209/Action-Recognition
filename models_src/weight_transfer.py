import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Transfers weights from one model to another by passing data into both models and comparing their output, whether they are the same are not.
Labels are binary, 1 if the input data was the same into each network and 0 otherwise.
avg_dim is dimension created from averaging outputs of each model.
"""
def Weight_Transfer(avg_dim=1024):
    model = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(avg_dim*2, avg_dim*2)),
        ('relu0', nn.ReLU(inplace=True)),
        ('fc1', nn.Linear(avg_dim*2, 512)),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(512, 128)),
        ('relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(128, 1)),
        ('out', nn.Sigmoid())
    ]))

    return model
