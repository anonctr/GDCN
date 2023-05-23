# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
import numpy as np
import torch
import torch.nn as nn

class FeaturesLinear(nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias

class FeaturesLinearWeight(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x, weight=None):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(torch.mul(self.fc(x), weight),dim=1) + self.bias

class FactorizationMachine(nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class FeaturesEmbedding1(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, initializer="xavier"):
        super().__init__()
        self.field_dims = field_dims
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims, embed_dim, concat=False):
        super().__init__()
        if isinstance(embed_dim, int):
            self.feature_embs = [embed_dim] * len(field_dims)
        elif isinstance(embed_dim, list):
            self.feature_embs = embed_dim
        self.feature_nums = field_dims
        self.embed_dict = nn.ModuleDict()
        self.feature_sum = sum(self.feature_nums)
        self.emb_sum = sum(self.feature_embs)
        self.len_field = len(self.feature_nums)
        self.concat = concat
        for field_index, (feature_num, feature_emb) in enumerate(zip(self.feature_nums, self.feature_embs)):
            embed = torch.nn.Embedding(feature_num, feature_emb)
            torch.nn.init.xavier_uniform_(embed.weight)
            self.embed_dict[str(field_index)] = embed

    def forward(self, x):
        sparse_embs = [self.embed_dict[str(index)](x[:, index]) for index in range(self.len_field)]

        if self.concat:
            # Independent dimension for DCN、DCN-V2 and GDCN
            sparse_embs = torch.cat(sparse_embs, dim=1)
            return sparse_embs
        # Equal dimension for most models
        sparse_embs = torch.stack(sparse_embs, dim=1)
        return sparse_embs


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, embed_dims, dropout=0.5, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim

        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)
        self._init_weight_()

    def _init_weight_(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        return self.mlp(x)


class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, field_dims, embed_dim, output_layer=True):
        super().__init__()
        self.num_fields = len(field_dims)
        #
        self.embeddings = torch.nn.ModuleList([torch.nn.Embedding(
            sum(field_dims), embed_dim) for _ in range(self.num_fields)])
        self.offsets = np.array(
            (0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

        self.output_layer = output_layer

    def forward(self, x):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]  # F,B,F,E
        ix = list()

        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])

        ix = torch.stack(ix, dim=1)
        if self.output_layer:
            ix = torch.sum(torch.sum(ix, dim=1), dim=1, keepdim=True)  # B,F,E => B,E => [B,1]
            return ix
        return ix



