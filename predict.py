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

from src.dataset import *
from src.test import test
from src.backbone import *
from src.util import *
import src.config as config

if __name__ == "__main__":
    args = config.parse_args()
    logger = logger(args)
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(device)

    backbone = C3D()
    net = Temp_Attn(args, device)
    if multi_gpus:
        backbone = nn.DataParallel(backbone).to(device)
        net = nn.DataParallel(net).to(device)
    else:
        backbone = backbone.to(device)
        net = net.to(device)
    if not args.load_backbone:
        backbone.load_state_dict(torch.load("models/c3d.pickle"))
    else:
        backbone.load_state_dict(torch.load(args.model_path.replace(".pth", "C3D.pth")))
    net.load_state_dict(torch.load(args.model_path))
    logger.name = os.path.basename(os.path.dirname(args.model_path))
    #testset = SegmentDataset(args.test_path, test=True)
    #testset = UCFCrime(test=True, target=args.target)
    #testset = UCFCrime_Fast(mode='test_resize', use_saliency=args.use_saliency, check=True, image_path=args.model_path)
    #testset = UCFCrime_Fast(mode='test_resize', use_saliency=args.use_saliency)
    testset = ShanghaiTech(mode='test_alt')
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    with torch.no_grad():
        result = test([backbone, net], testloader, device, args, logger)
    try:
        roc, roc_bag = result.roccurve()
        logger.savefig(roc, "ROC.png")
        logger.savefig(roc_bag, "BagROC.png")
    except:
        ipdb.set_trace()
    #logger.auc_types(result)
    logger.recordauc(result, 0)
