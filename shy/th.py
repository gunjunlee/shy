import torch
import torch.nn as nn

import time
import copy
import os


def safe_save_net(net):
    if hasattr(net, 'module'):
        return net.module.state_dict()
    return net.state_dict()


def safe_load_net(net, state_dict):
    if hasattr(net, 'module'):
        net.module.load_state_dict(state_dict)
    else:
        net.load_state_dict(state_dict)
    return net


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
