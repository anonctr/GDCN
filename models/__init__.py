# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""
import sys

import torch
from torch import nn

sys.path.append("..")

from models.LR import LogisticRegression
from models.FM import FactorizationMachineModel
from models.DeepFM import DeepFM
from models.FFM import FieldAwareFactorizationMachineModel
from models.FwFM import FwFM
from models.FmFM import FMFM

from models.DCN import DCN, CrossNetwork
from models.GDCN import GDCNS, GDCNP, GateCorssNetwork
from models.DCNV2 import DCNV2S, DCNV2P, CrossNetV2
from models.MaskNet import MaskNet
from models.PNN import IPNN, OPNN
from models.AFN import AFN, AFNPlus
from models.FNN import FNN
from models.AutoInt import AutoInt, AutoIntPlus
from models.xDeepFM import CIN, xDeepFM
from models.WDL import WideAndDeep
from models.FED import FED
from models.NON import NON

if __name__ == '__main__':
    import numpy as np

    fd = [3, 4]
    embed_dim = 8
    f_n = np.array([[1, 3], [0, 2], [0, 1], [1, 3]])
    f_n = torch.from_numpy(f_n).long()
    model = FNN(fd, embed_dim)

    print(model)
    label = torch.randint(0, 2, (4, 1)).float()
    print(label)
    loss = nn.BCEWithLogitsLoss()
    pred = model(f_n)
    print(pred.size())
    losses = loss(pred, label)
    print(losses)
