# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""


import torch
import torch.nn as nn
from models.basic_layers import  FeaturesEmbedding, MultiLayerPerceptron


class FNN(nn.Module):
    def __init__(self, field_dims, embed_dim, num_layers=3, dropout=0.5):
        super(FNN, self).__init__()
        mlp_layers = [400] * num_layers
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, concat=True)
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        else:
            self.embed_output_dim = sum(embed_dim)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, embed_dims=mlp_layers,
                                        dropout=dropout, output_layer=True)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.mlp(x_emb.view(x.size(0), -1))
        return pred_y
