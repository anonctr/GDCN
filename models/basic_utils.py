# -*- coding: UTF-8 -*-
"""
@project:CTR
@time:2022/9/21 11:08
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os

def get_activation(activation):
    if isinstance(activation, str):
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "prelu":
            return nn.PReLU()
        elif activation.lower() == "linear":
            return nn.Identity()
        else:
            return getattr(nn, activation)()
    else:
        return activation
