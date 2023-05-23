# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.basic_layers import FeaturesEmbedding, MultiLayerPerceptron
from models.basic_utils import get_activation


class MaskNet(nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_layers=[400, 400, 400], block_units=[400, 400, 400],
                 dropout=0.2, model_type="SerialMaskNet", dnn_hidden_activations="ReLU", parallel_num_blocks=3,
                 parallel_block_dim=64, reduction_ratio=1, emb_layernorm=True, net_layernorm=True):
        super(MaskNet, self).__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.num_fields = len(field_dims)
        self.input_dim = self.num_fields * embed_dim
        if model_type == "SerialMaskNet":
            self.mask_net = SerialMaskNet(input_dim=self.input_dim,
                                          block_units=block_units,
                                          hidden_activations=dnn_hidden_activations,
                                          reduction_ratio=reduction_ratio,
                                          dropout=dropout,
                                          layer_norm=net_layernorm)

        elif model_type == "ParallelMaskNet":
            self.mask_net = ParallelMaskNet(self.input_dim,
                                            num_blocks=parallel_num_blocks,
                                            block_dim=parallel_block_dim,
                                            mlp_layers=mlp_layers,
                                            reduction_ratio=reduction_ratio,
                                            dropout=dropout,
                                            layer_norm=net_layernorm)
        if emb_layernorm:
            self.emb_norm = nn.LayerNorm([self.num_fields, embed_dim])
        else:
            self.emb_norm = None

    def forward(self, x):
        x_emb = self.embedding(x)
        if self.emb_norm:
            x_emb = self.emb_norm(x_emb)
        pred_y = self.mask_net(x_emb.flatten(start_dim=1))
        return pred_y


class ParallelMaskNet(nn.Module):
    def __init__(self, input_dim, num_blocks=1, block_dim=64,
                 mlp_layers=[400,400,400], hidden_activations="ReLU",
                 reduction_ratio=1, dropout=0.2,
                 layer_norm=True):
        super(ParallelMaskNet, self).__init__()
        self.num_blocks = num_blocks
        self.mask_blocks = nn.ModuleList([MaskBlock(input_dim,
                                                    input_dim,
                                                    block_dim,
                                                    hidden_activations,
                                                    reduction_ratio,
                                                    dropout_rate=dropout,
                                                    layer_norm=layer_norm) for _ in range(num_blocks)])

        self.mlp = MultiLayerPerceptron(block_dim * num_blocks, mlp_layers,
                                        output_layer=True, dropout=dropout)

    def forward(self, x_cat):
        block_out = []
        for i in range(self.num_blocks):
            block_out.append(self.mask_blocks[i](x_cat, x_cat))
        concat_out = torch.cat(block_out, dim=-1)
        v_out = self.mlp(concat_out)
        return v_out


class SerialMaskNet(nn.Module):
    def __init__(self, input_dim, block_units=[400, 400, 400],
                 hidden_activations="ReLU", reduction_ratio=1,
                 dropout=0.0, layer_norm=True):
        super(SerialMaskNet, self).__init__()
        self.hidden_units = [input_dim] + block_units
        self.mask_blocks = nn.ModuleList()
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(MaskBlock(input_dim,
                                              self.hidden_units[idx],
                                              self.hidden_units[idx + 1],
                                              hidden_activations,
                                              reduction_ratio,
                                              dropout,
                                              layer_norm))
        self.fc = nn.Linear(self.hidden_units[-1], 1)

    def forward(self, x_cat):
        v_out = x_cat
        for idx in range(len(self.hidden_units) - 1):
            v_out = self.mask_blocks[idx](x_cat, v_out)
        y_pred = self.fc(v_out)
        return y_pred


class MaskBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_activation="ReLU",
                 reduction_ratio=1, dropout_rate=0.2, layer_norm=True):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim * reduction_ratio),
                                        nn.ReLU(),
                                        nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim))

        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(get_activation(hidden_activation))
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, x, h):
        v_mask = self.mask_layer(x)
        v_out = self.hidden_layer(v_mask * h)
        return v_out