# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

#  The list of field features
all_field_dims = {
    "criteo": [49, 101, 126, 45, 223, 118, 84, 76, 95, 9,
               30, 40, 75, 1458, 555, 193949, 138801, 306, 19, 11970,
               634, 4, 42646, 5178, 192773, 3175, 27, 11422, 181075, 11,
               4654, 2032, 5, 189657, 18, 16, 59697, 86, 45571],
}
data_name = "criteo"
field_dims = all_field_dims[data_name]
# Load the trained model
model_path = "./chkpts/criteo/gcn3_best_auc_16_0228193156.pkl"
models = torch.load(model_path, map_location="cpu")


def get_field_embeddings(model, field_num):
    field_dims = all_field_dims[data_name]
    cum_sum = np.array((0, *np.cumsum(field_dims)), dtype=np.compat.long)
    start = cum_sum[field_num]
    end = cum_sum[field_num + 1]
    embeddings = model.embedding.embedding.weight[start:end]
    return embeddings.detach().numpy()


def calcute_dimension(model=None, ratio=0.95, num_field=4):
    record = []
    pca = PCA(n_components=ratio)
    for i in range(0, num_field):
        field_index = i
        embeddings_field0 = get_field_embeddings(model, field_index)
        pca.fit(embeddings_field0)
        print(i, "---", pca.n_components_)
        record.append(pca.n_components_)
    return record


records = defaultdict(list)
for ratio in np.arange(0.5, 1.0, 0.05):
    record = calcute_dimension(models, ratio, num_field=39)
    records[ratio] = record

# Save the dimension
records_df = pd.DataFrame(records)
records_df["field_num"] = all_field_dims
records_df.to_excel("fdo_dims.xlsx", sheet_name="more_data")
