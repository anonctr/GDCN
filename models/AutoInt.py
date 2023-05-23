# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""


import torch
import torch.nn.functional as F

from models.basic_layers import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class AutoIntPlus(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, atten_embed_dim=64, num_heads=2,
                 num_layers=3, mlp_dims=(400, 400, 400), dropouts=(0.5,0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)
        pred_y = self.linear(x) + self.attn_fc(cross_term) + self.mlp(x_emb.view(-1, self.embed_output_dim))
        return pred_y

class AutoInt(torch.nn.Module):
    """
    A pytorch implementation of AutoInt.

    Reference:
        W Song, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks, 2018.
    """

    def __init__(self, field_dims, embed_dim, atten_embed_dim=128, num_heads=8,
                 num_layers=3, dropouts=(0.5, 0.5), has_residual=True):
        super().__init__()
        self.num_fields = len(field_dims)
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.atten_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)
        # self.embed_output_dim = len(field_dims) * embed_dim
        self.atten_output_dim = len(field_dims) * atten_embed_dim
        self.has_residual = has_residual
        # self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropouts[1])

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(atten_embed_dim, num_heads, dropout=dropouts[0]) for _ in range(num_layers)
        ])
        self.attn_fc = torch.nn.Linear(self.atten_output_dim, 1)
        if self.has_residual:
            self.V_res_embedding = torch.nn.Linear(embed_dim, atten_embed_dim)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x_emb = self.embedding(x)
        atten_x = self.atten_embedding(x_emb)

        cross_term = atten_x.transpose(0, 1)

        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        if self.has_residual:
            V_res = self.V_res_embedding(x_emb)
            cross_term += V_res

        cross_term = F.relu(cross_term).contiguous().view(-1, self.atten_output_dim)

        pred_y = self.attn_fc(cross_term) + self.linear(x)
        return pred_y
