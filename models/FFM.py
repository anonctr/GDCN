# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""


import torch
import torch.nn as nn
from models.basic_layers import FeaturesLinear, FieldAwareFactorizationMachine


class FieldAwareFactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        pred_y = self.linear(x) + ffm_term
        return pred_y