# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_layers import FeaturesEmbedding, MultiLayerPerceptron, FactorizationMachine

class FED(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), drm_flag=False, dropout=0.5):
        super(FED, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims=field_dims, embed_dim=embed_dim)
        self.drm_flag = drm_flag
        self.drm = DRM(len(field_dims))
        self.mlp_input_dim = len(field_dims) * embed_dim
        self.element_wise = MultiLayerPerceptron(self.mlp_input_dim, mlp_layers, output_layer=False)

        self.field_wise = FieldAttentionModule(embed_dim)

        self.out_len = self.mlp_input_dim * 2 + list(mlp_layers)[-1]

        self.fm = FactorizationMachine(reduce_sum=True)

        self.lin_out = nn.Linear(self.out_len, 1)

    def forward(self, x):
        b = x.size(0)
        E = self.embedding(x)
        if self.drm_flag:
            E = self.drm(E)
        E_f = self.field_wise(E) + E
        E_e = self.element_wise(E.reshape(b, -1))
        E_con = torch.cat([E_f.reshape(b, -1),
                           E_e.reshape(b, -1),
                           E.reshape(b, -1)], dim=1)
        pred_y = self.lin_out(E_con)
        return pred_y

class DRM(nn.Module):
    def __init__(self, num_field):
        super(DRM, self).__init__()
        self.fan = FieldAttentionModule(num_field)

    def forward(self, V):
        U = V.permute(0, 2, 1)  # B,E,F
        E = self.fan(U).permute(0, 2, 1)  # B,F,E
        E = E + V
        return E

class FieldAttentionModule(nn.Module):
    def __init__(self, embed_dim):
        super(FieldAttentionModule, self).__init__()
        self.trans_Q = nn.Linear(embed_dim, embed_dim)
        self.trans_K = nn.Linear(embed_dim, embed_dim)
        self.trans_V = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, scale=None):
        Q = self.trans_Q(x)
        K = self.trans_K(x)
        V = self.trans_V(x)

        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)

        return context