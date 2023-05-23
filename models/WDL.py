# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""

import torch
import torch.nn as nn
from models.basic_layers import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron

class WideAndDeep(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(WideAndDeep, self).__init__()
        self.lr = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.lr(x) + self.mlp(x_emb.view(x.size(0), -1))
        return pred_y