# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
import torch
import torch.nn as nn
from models.basic_layers import FeaturesEmbedding, MultiLayerPerceptron

class DCN(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(DCN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, concat=True)
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        elif isinstance(embed_dim, list):
            self.embed_output_dim = np.sum(embed_dim)
        self.cross_net = CrossNetwork(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        print(x_emb.size())
        # x_emb = x_emb.reshape(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)
        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y

class CN(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(CN, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, concat=True)
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        elif isinstance(embed_dim, list):
            self.embed_output_dim = np.sum(embed_dim)
        self.cross_net = CrossNetwork(self.embed_output_dim, cn_layers)
        self.fc = nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        cross_cn = self.cross_net(x_emb)
        pred_y = self.fc(cross_cn)
        return pred_y


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, cn_layers):
        super().__init__()
        self.cn_layers = cn_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(cn_layers)
        ])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.cn_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x

