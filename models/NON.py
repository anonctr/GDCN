# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_layers import FeaturesLinear, FeaturesEmbedding, MultiLayerPerceptron

class NON(nn.Module):
    def __init__(self, field_dims, embed_dim, att_size=64, mlp_layers=(400, 400, 400), dropouts=(0.5, 0.5)):
        super(NON, self).__init__()

        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.non_enhance = NONLayer(len(field_dims), embed_dim)

        self.dnn = MultiLayerPerceptron(embed_dim * len(field_dims), mlp_layers, dropout=dropouts[0],
                                        output_layer=False)

        self.att_embedding = torch.nn.Linear(embed_dim, att_size)
        self.att_output_dim = len(field_dims) * att_size

        self.self_attns = torch.nn.ModuleList([
            torch.nn.MultiheadAttention(att_size, 2, dropout=dropouts[0]) for _ in range(3)
        ])

        self.input_dim = 400 + self.att_output_dim + 1
        self.mlp = MultiLayerPerceptron(self.input_dim, embed_dims=(64, 32), dropout=dropouts[1])

    def forward(self, x):
        x_emb = self.embedding(x)
        x_emb, _ = self.non_enhance(x_emb)
        x_fc = self.linear(x)
        x_dnn = self.dnn(x_emb.reshape(x.size(0), -1))

        att_x = self.att_embedding(x_emb)
        cross_term = att_x.transpose(0, 1)
        for self_attn in self.self_attns:
            cross_term, _ = self_attn(cross_term, cross_term, cross_term)

        cross_term = cross_term.transpose(0, 1)
        cross_term = F.relu(cross_term).contiguous().view(-1, self.att_output_dim)

        x_final = torch.cat([x_fc,
                             cross_term.view(x.size(0), -1),
                             x_dnn.view(x.size(0), -1)],
                            dim=1)

        pred_y = self.mlp(x_final)
        return pred_y

class NONLayer(nn.Module):
    def __init__(self, field_length, embed_dim):
        super(NONLayer, self).__init__()
        self.input_dim = field_length * embed_dim
        self.local_w = nn.Parameter(torch.randn(field_length, embed_dim, embed_dim))
        self.local_b = nn.Parameter(torch.randn(field_length, 1, embed_dim))
        nn.init.xavier_uniform_(self.local_w.data)
        nn.init.xavier_uniform_(self.local_b.data)

    def forward(self, x_emb):
        x_local = torch.matmul(x_emb.permute(1, 0, 2), self.local_w) + self.local_b
        x_local0 = torch.relu(x_local).permute(1, 0, 2)
        # F（ei，ei`）
        x_local = x_local0 * x_emb
        return x_local.contiguous(), x_local0