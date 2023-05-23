# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""

import torch
import torch.nn as nn
from models.basic_layers import FeaturesEmbedding, MultiLayerPerceptron


class IPNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5):
        super(IPNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.pnn = InnerProductNetwork(num_fields)

        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout=dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        cross_ipnn = self.pnn(x_emb)

        x = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_ipnn], dim=1)
        pred_y = self.mlp(x)
        return pred_y


class OPNN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, kernel_type="vec"):
        super(OPNN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        num_fields = len(field_dims)
        self.pnn = OuterProductNetwork(num_fields, embed_dim, kernel_type)
        self.embed_output_dim = num_fields * embed_dim
        self.inter_size = num_fields * (num_fields - 1) // 2
        self.mlp = MultiLayerPerceptron(self.inter_size + self.embed_output_dim, mlp_layers, dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        cross_opnn = self.pnn(x_emb)
        x = torch.cat([x_emb.view(-1, self.embed_output_dim), cross_opnn], dim=1)
        pred_y = self.mlp(x)
        return pred_y


class InnerProductNetwork(nn.Module):
    def __init__(self, num_fields):
        super(InnerProductNetwork, self).__init__()
        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        return torch.sum(x[:, self.row] * x[:, self.col], dim=2)


class OuterProductNetwork(nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

        self.row, self.col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row.append(i), self.col.append(j)

    def forward(self, x):
        p, q = x[:, self.row], x[:, self.col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)  # b,num_ix,e
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)
