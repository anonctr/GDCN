# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_layers import FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class CIN(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(400,400,400), dropout=0.5,
                 cross_layer_sizes=(100, 100), split_half=False):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

    def forward(self, x):
        x_emb = self.embedding(x)
        pred_y = self.cin(x_emb)
        return pred_y

class xDeepFM(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(400, 400, 400), dropout=0.5,
                 cross_layer_sizes=(50, 50), split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        x_emb = self.embedding(x)

        lr_term = self.linear(x)
        cin_term = self.cin(x_emb)
        mlp_term = self.mlp(x_emb.view(-1, self.embed_output_dim))

        pred_y =  cin_term + mlp_term + lr_term
        return pred_y

class CompressedInteractionNetwork(nn.Module):
    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))