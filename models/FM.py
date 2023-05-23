# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""

import torch.nn as nn
from models.basic_layers import FeaturesLinear, FeaturesEmbedding, FactorizationMachine

class FactorizationMachineModel(nn.Module):
    def __init__(self, field_dims, emb_dim):
        super(FactorizationMachineModel, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, emb_dim)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.lr(x) + self.fm(x_emb)
        return pred_y


