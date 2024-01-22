# -*- coding: UTF-8 -*-
"""
@project:GDCNv2
first: a^4=a^2**2=a^2*a^2
second: mask operation only use in the last layer 
"""

import torch
import math
import torch.nn as nn

from models.basic_layers import FeaturesEmbedding, MultiLayerPerceptron


class GDCNv2P(nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(GDCNP, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, concat=True)
        # self.embed_output_dim = len(field_dims) * embed_dim
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        else:
            self.embed_output_dim = sum(embed_dim)
        self.cross_net = GateCorssLayerv2(self.embed_output_dim, cn_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=False, dropout=dropout)
        self.fc = torch.nn.Linear(mlp_layers[-1] + self.embed_output_dim, 1)

    def forward(self, x):
        x_emb = self.embedding(x)
        # x_emb = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_emb)
        cross_mlp = self.mlp(x_emb)
        pred_y = self.fc(torch.cat([cross_cn, cross_mlp], dim=1))
        return pred_y


class GDCNv2S(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3, mlp_layers=(400, 400, 400), dropout=0.5):
        super(GDCNS, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, concat=True)
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        else:
            self.embed_output_dim = sum(embed_dim)
        self.cross_net = GateCorssLayerv2(self.embed_output_dim, cn_layers)
        self.pred_layer = MultiLayerPerceptron(self.embed_output_dim, mlp_layers, output_layer=True,
                                               dropout=dropout)

    def forward(self, x):
        x_embed = self.embedding(x)
        # x_embed = self.embedding(x).view(-1, self.embed_output_dim)
        cross_cn = self.cross_net(x_embed)
        pred_y = self.pred_layer(cross_cn)
        return pred_y


class GateCorssNetwork(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, cn_layers=3):
        super(GateCorssNetwork, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim, concat=True)
        if isinstance(embed_dim, int):
            self.embed_output_dim = len(field_dims) * embed_dim
        else:
            self.embed_output_dim = sum(embed_dim)
        self.cross_net = GateCorssLayerv2(self.embed_output_dim, cn_layers)
        self.pred_layer = torch.nn.Linear(self.embed_output_dim, 1)

    def forward(self, x):
        x_embed = self.embedding(x)
        cross_cn = self.cross_net(x_embed)
        pred_y = self.pred_layer(cross_cn)
        return pred_y


class GateCorssLayerv2(nn.Module):
    #  The core structure： gated corss layer.
    def __init__(self, input_dim, cn_layers=3):
        super().__init__()

        # self.cn_layers = cn_layers
        self.cn_laysers = int(math.log(cn_layers,2))
        self.flag = True if math.ceil(math.log(cn_layers,2))>self.cn_layers else False
        if self.flag:
            self.cn_layers += 1
        
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])
        self.wg = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(cn_layers)
        ])

        self.b = torch.nn.ParameterList([torch.nn.Parameter(
            torch.zeros((input_dim,))) for _ in range(cn_layers)])

        for i in range(cn_layers):
            torch.nn.init.uniform_(self.b[i].data)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        xw = self.w[0](x) # Feature Crossing
        xg = self.activation(self.wg[0](x)) # Information Gate
        x = xw + self.b[0] 
        
        for i in range(1,self.cn_layers):
            x = x * x  + x
        
        if self.flag:
            xw = self.w[-1](x) # Feature Crossing
            xg = self.activation(self.wg[-1](x)) # Information Gate
            x = x *(xw + self.b[-1])*xg + x
            
        return x


if __name__ == '__main__':
    import numpy as np

    fd = [3, 4]
    embed_dim = [3,10]
    # embed_dim = 8
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()
    model = GateCorssNetwork(fd, embed_dim)

    print(model)
    label = torch.randint(0, 2, (4, 1)).float()
    print(label)
    loss = nn.BCEWithLogitsLoss()
    pred = model(f_n)
    print(pred.size())
    losses = loss(pred, label)
    print(losses)
