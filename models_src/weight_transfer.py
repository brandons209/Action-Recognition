import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
Transfers weights from one model to another by passing data into both models and comparing their output, whether they are the same are not.
Labels are binary, 1 if the input data was the same into each network and 0 otherwise.
avg_dim is dimension of the last linear layer of each model to average the outputs.
The 3D network should not have a classifier, last output should be output of last convolutional layer in network
"""
def weight_transfer_fc(inp_dim):
    #these might work better with leaky relu.
    model = nn.Sequential(OrderedDict([

        ('fc1', nn.Linear(inp_dim, inp_dim//4)),
        ('relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(inp_dim//4, inp_dim//16)),
        ('relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(inp_dim//16, 1)),
        ('out', nn.Sigmoid())
    ]))

    return model

"""
inputs should be the 2d and 3d models, with the T3D model not having its classifier on.
Also needs the output features of each model, the T3D model has a function to get those, and denset2D has 1000 feature output
"""
class Weight_Transfer(nn.Module):
    def __init__(self, twoD_model, threeD_model, twoD_out_features, threeD_out_features, frames_per_batch, avg_dim=1024):
        super(Weight_Transfer, self).__init__()
        self.twoD_model = twoD_model
        self.threeD_model = threeD_model
        self.inp_dim = avg_dim * 2
        self.transfer = weight_transfer_fc(self.inp_dim)
        self.num_frames = frames_per_batch

        #transform layers to transform outputs of 2D and 3D networks to an average dimension size for weight transfer
        self.twoD_transform_layer = nn.Linear(twoD_out_features, avg_dim)
        self.threeD_transform_layer = nn.Linear(threeD_out_features, avg_dim)

        #set 2D model to not trainable.
        for param in twoD_model.parameters():
            param.requires_grad = False

    def forward(self, x):#x should be a sequence of frames
        write = 1
        twoD_out = None
        for i in range(self.num_frames):#add preds from 2D model accross frames
            if write:
                x_farme = x[i].unsqueeze(0)
                twoD_out = self.twoD_model(x_farme)
                write = 0
            else:
                x_farme = x[i].unsqueeze(0)
                twoD_out += self.twoD_model(x_farme)

        twoD_out = twoD_out / x.size(0)#average output from 2d model

        try:#if this works, another set of frames is in x for negative pair, which will go into the 3D model
            sh = x[self.num_frames:].shape
            print(sh)
            x_farme = x[self.num_frames:].unsqueeze(0).view(1, sh[1], sh[0], sh[2], sh[3])
            threeD_out = self.threeD_model(x_farme)
        except:#not a negative pair
            #looks like output is already flattened for T3D model
            sh = x.shape
            x_farme = x.unsqueeze(0).view(1, sh[1], sh[0], sh[2], sh[3])
            threeD_out = self.threeD_model(x_farme)

        #transform outputs into average dimesions
        twoD_out = self.twoD_transform_layer(twoD_out)
        threeD_out = self.threeD_transform_layer(threeD_out)

        #concat outputs and passthrough to weight transfer FC layers
        concat = torch.cat([twoD_out, threeD_out], 1)
        pred = self.transfer(concat)
        return pred
