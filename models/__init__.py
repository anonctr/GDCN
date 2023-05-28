# -*- coding: UTF-8 -*-
"""
@project:GDCN
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
