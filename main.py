import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim

import csv
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import ipdb

from src.dataset import FrameFolderDataset, SegmentDataset
from src.pytorch_i3d import InceptionI3d
from src.backbone import C3D, Attention
from src.predict import predict
from src.train import train
import src.util as util
import src.config as config
        
def main():
    args = config.parse_args()
    logger = util.logger(args)

    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(device)

    backbone = C3D()
    backbone.load_state_dict(torch.load("models/c3d.pickle"))
    net = Attention(args.attention_type)
    if multi_gpus:
        backbone = nn.DataParallel(backbone).to(device)
        net = nn.DataParallel(net).to(device)
    else:
        backbone = backbone.to(device)
        net = net.to(device)

    trainset = SegmentDataset(args.train_path)
    trainloader = DataLoader(trainset, batch_size = 1, shuffle=True)
        
    testset = SegmentDataset(args.test_path, test=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

    maxauc = 0
    for epoch in range(args.epoch):
        logger.info("Epoch: {}/{}".format(epoch, args.epoch))
        net, losses = train(net, trainloader, device, optimizer)
        logger.recordloss(losses, epoch)

        result = predict(net, testloader, device, args)
        logger.recordauc(result, epoch)        

        if result.auc() > maxauc:
            maxauc = result.auc()
            logger.savemodel(net, epoch)
    logger.info("best preformance: {:.4f}".format(maxauc))

        

if __name__ == '__main__':
    main()
