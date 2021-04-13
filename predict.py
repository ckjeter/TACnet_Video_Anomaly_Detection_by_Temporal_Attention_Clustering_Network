import os
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

import csv
import numpy as np
import logging
import ipdb

from src.dataset import FrameFolderDataset, SegmentDataset
from src.test import test
from src.backbone import C3D, Attention
from src.util import Averager, Scorer, AnomalyResult
import src.util as util
import src.config as config

if __name__ == "__main__":
    args = config.parse_args()
    logger = util.logger(args)
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(device)

    backbone = C3D()
    backbone.load_state_dict(torch.load("models/c3d.pickle"))
    net = Attention(args, device)
    if multi_gpus:
        backbone = nn.DataParallel(backbone).to(device)
        net = nn.DataParallel(net).to(device)
    else:
        backbone = backbone.to(device)
        net = net.to(device)
        
    testset = SegmentDataset(args.test_path, test=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    net.load_state_dict(torch.load(args.model_path))
    result = test(net, testloader, device, args)
    roc = result.roccurve()
    roc.savefig("ROC.png")
    logger.auc_types(result)
    print(result.auc())
