# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.basic_layers import FeaturesLinear, FactorizationMachine, MultiLayerPerceptron, FeaturesEmbedding


class FMFM(nn.Module):
    def __init__(self, field_dims, embed_dim, interaction_type="matrix"):
        super(FMFM, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        # self.lr = FeaturesLinear(field_dims)
        self.num_field = len(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.inter_num = self.num_field * (self.num_field - 1) // 2
        self.field_interaction_type = interaction_type
        if self.field_interaction_type == "vector":  # FvFM
            # I,E
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim))
        elif self.field_interaction_type == "matrix":  # FmFM
            # I,E,E
            self.interaction_weight = nn.Parameter(torch.Tensor(self.inter_num, embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.interaction_weight.data)
        self.row, self.col = list(), list()
        for i in range(self.num_field - 1):
            for j in range(i + 1, self.num_field):
                self.row.append(i), self.col.append(j)

    def forward(self, x):

        x_emb = self.embedding(x)
        left_emb = x_emb[:, self.row]
        right_emb = x_emb[:, self.col]
        if self.field_interaction_type == "vector":
            left_emb = left_emb * self.interaction_weight
        elif self.field_interaction_type == "matrix":
            # B,I,1,E * I,E,E = B,I,1,E => B,I,E
            left_emb = torch.matmul(left_emb.unsqueeze(2), self.interaction_weight).squeeze(2)
        pred_y = (left_emb * right_emb).sum(dim=-1).sum(dim=-1, keepdim=True)
        return pred_y
