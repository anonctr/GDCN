# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""

import itertools
import torch
import torch.nn as nn

from models.basic_layers import FeaturesEmbedding, MultiLayerPerceptron

class FiBiNet(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=(400, 400, 400), dropout=0.5, bilinear_type="all"):
        super(FiBiNet, self).__init__()
        num_fields = len(field_dims)

        self.embedding = FeaturesEmbedding(field_dims, embed_dim)

        self.senet = SenetLayer(num_fields)

        self.bilinear = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)
        self.bilinear2 = BilinearInteractionLayer(num_fields, embed_dim, bilinear_type=bilinear_type)

        num_inter = num_fields * (num_fields - 1) // 2
        self.embed_output_size = num_inter * embed_dim
        self.mlp = MultiLayerPerceptron(2 * self.embed_output_size, mlp_layers, dropout=dropout)

    def forward(self, x):
        x_emb = self.embedding(x)
        x_senet, x_weight = self.senet(x_emb)

        x_bi1 = self.bilinear(x_emb)
        x_bi2 = self.bilinear2(x_senet)

        x_con = torch.cat([x_bi1.view(x.size(0), -1),
                           x_bi2.view(x.size(0), -1)], dim=1)

        pred_y = self.mlp(x_con)
        return pred_y

class SenetLayer(nn.Module):
    def __init__(self, field_length, ratio=1):
        super(SenetLayer, self).__init__()
        self.temp_dim = max(1, field_length // ratio)
        self.excitation = nn.Sequential(
            nn.Linear(field_length, self.temp_dim),
            nn.ReLU(),
            nn.Linear(self.temp_dim, field_length),
            nn.ReLU()
        )

    def forward(self, x_emb):
        Z_mean = torch.max(x_emb, dim=2, keepdim=True)[0].transpose(1, 2)
        # Z_mean = torch.mean(x_emb, dim=2, keepdim=True).transpose(1, 2)
        A_weight = self.excitation(Z_mean).transpose(1, 2)
        V_embed = torch.mul(A_weight, x_emb)
        return V_embed, A_weight

class BilinearInteractionLayer(nn.Module):
    def __init__(self, filed_size, embedding_size, bilinear_type="interaction"):
        super(BilinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        self.bilinear = nn.ModuleList()

        if self.bilinear_type == "all":
            self.bilinear = nn.Linear(
                embedding_size, embedding_size, bias=False)

        elif self.bilinear_type == "each":
            for i in range(filed_size):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))

        elif self.bilinear_type == "interaction":
            for i, j in itertools.combinations(range(filed_size), 2):
                self.bilinear.append(
                    nn.Linear(embedding_size, embedding_size, bias=False))
        else:
            raise NotImplementedError

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        inputs = torch.split(inputs, 1, dim=1)
        if self.bilinear_type == "all":
            p = [torch.mul(self.bilinear(v_i), v_j)
                 for v_i, v_j in itertools.combinations(inputs, 2)]

        elif self.bilinear_type == "each":
            p = [torch.mul(self.bilinear[i](inputs[i]), inputs[j])
                 for i, j in itertools.combinations(range(len(inputs)), 2)]

        elif self.bilinear_type == "interaction":
            p = [torch.mul(bilinear(v[0]), v[1])
                 for v, bilinear in zip(itertools.combinations(inputs, 2), self.bilinear)]
        else:
            raise NotImplementedError
        return torch.cat(p, dim=1)