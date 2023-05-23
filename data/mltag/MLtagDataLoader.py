# -*- coding: UTF-8 -*-
"""
@project:GDCN
"""
import pandas as pd
import torch

class LoadData811():
    def __init__(self, path="./Data/", dataset="mltag"):
        self.dataset = dataset
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + ".train.libfm"
        self.testfile = self.path + dataset + ".test.libfm"
        self.validationfile = self.path + dataset + ".validation.libfm"
        self.features_M = {}
        self.construct_df()

    def construct_df(self):
        self.data_train = pd.read_table(self.trainfile, sep=" ", header=None, engine='python')
        self.data_test = pd.read_table(self.testfile, sep=" ", header=None, engine="python")
        self.data_valid = pd.read_table(self.validationfile, sep=" ", header=None, engine="python")

        for i in self.data_test.columns[1:]:
            self.data_test[i] = self.data_test[i].apply(lambda x: int(x.split(":")[0]))
            self.data_train[i] = self.data_train[i].apply(lambda x: int(x.split(":")[0]))
            self.data_valid[i] = self.data_valid[i].apply(lambda x: int(x.split(":")[0]))

        self.all_data = pd.concat([self.data_train, self.data_test, self.data_valid])
        self.field_dims = []

        for i in self.all_data.columns[1:]:
            maps = {val: k for k, val in enumerate(set(self.all_data[i]))}
            self.all_data[i] = self.all_data[i].map(maps)
            self.features_M[i] = maps
            self.field_dims.append(len(set(self.all_data[i])))
        self.all_data[0] = self.all_data[0].apply(lambda x: max(x, 0))


class RecData(torch.utils.data.Dataset):
    def __init__(self, all_data):
        self.data_df = all_data

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        x = self.data_df.iloc[idx].values[1:]
        y = self.data_df.iloc[idx].values[0]
        return x, y

def get_mltag_dataloader811(path="../data/", dataset="mltag", num_ng=4, batch_size=4096):
    AllDataF = LoadData811(path=path, dataset=dataset)
    all_dataset = RecData(AllDataF.all_data)

    train_size = int(0.9 * len(all_dataset))
    test_size = len(all_dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size - test_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_ng, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=num_ng)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_ng)
    print(len(train_loader))
    print(len(valid_loader))
    print(len(test_loader))
    return AllDataF.field_dims, train_loader, valid_loader, test_loader