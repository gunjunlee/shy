import torch
import torch.nn as nn
from torchvision import transforms

import time
import copy
import os

imagenet_stat = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

def norm2bgr(image, mean=imagenet_stat['mean'], std=imagenet_stat['std']):
    temp = transforms.functional.normalize(image, mean=[0, 0, 0], std=[1/std[0], 1/std[1], 1/std[2]])
    temp = transforms.functional.normalize(temp, mean=[-mean[0], -mean[1], -mean[2]], std=[1, 1, 1])
    temp = (temp*255).byte()
    temp = temp.permute(1, 2, 0)
    return temp

def bgr2norm(image, mean=imagenet_stat['mean'], std=imagenet_stat['std']):
    temp = transforms.functional.to_tensor(image)
    temp = transforms.functional.normalize(image, mean=mean, std=std)
    return temp

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
