import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Transfers weights from one model to another by passing data into both models and comparing their output, whether they are the same are not.
Labels are binary, 1 if the input data was the same into each network and 0 otherwise.
avg_dim is dimension created from averaging outputs of each model.
"""
def weight_transfer_fc(avg_dim):
    inp_dim = avg_dim * 2
    model = nn.Sequential(OrderedDict([
        ('fc0', nn.Linear(inp_dim, inp_dim)),
        ('relu0', nn.ReLU(inplace=True)),
        ('fc1', nn.Linear(inp_dim, inp_dim/4)),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(inp_dim/4, inp_dim/16)),
        ('relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(inp_dim/16, 1)),
        ('out', nn.Sigmoid())
    ]))

    return model


class Weight_Transfer(nn.Module):
    def __init(self, twoD_model, threeD_model, avg_dim=1024):
        self.twoD_model = twoD_model
        self.threeD_model = threeD_model
        self.inp_dim = avg_dim * 2

        for param in twoD_model.parameters():
            param.requires_grad = False

        def forward(self, x):
            twoD_out = None
            for frame in x:
                pred = self.twoD_model(x)
