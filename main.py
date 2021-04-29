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

from src.dataset import FrameFolderDataset, SegmentDataset, UCFCrime
from src.pytorch_i3d import InceptionI3d
from src.backbone import C3D, Attention
from src.test import test
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
    logger.info("Device: {}".format(device))

    #----------Prepare Datasets----------
    #trainset = SegmentDataset(args.train_path)
    trainset = UCFCrime(test=False, target = args.target)
    trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle=True)
        
    #testset = SegmentDataset(args.test_path, test=True)
    testset = UCFCrime(test=True, target = args.target)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)
    
    #----------Prepare Models----------
    backbone = C3D()
    net = Attention(args, device)
    if len(args.model_path) > 0:
        backbone.load_state_dict(torch.load(args.model_path.replace(".pth", "C3D.pth")))
        net.load_state_dict(torch.load(args.model_path))
        epoch_start = int(os.path.basename(args.model_path).split(".")[0])
    else:
        backbone.load_state_dict(torch.load("models/c3d.pickle"))
        epoch_start = 0
    
    if multi_gpus:
        backbone = nn.DataParallel(backbone).to(device)
        net = nn.DataParallel(net).to(device)
    else:
        backbone = backbone.to(device)
        net = net.to(device)
    
    #----------Prepare Training Shit----------
    optimizer = optim.AdamW([
        {'params': backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': net.parameters(), 'lr': args.lr}
    ])
    #optimizer = optim.SGD(net.parameters(), momentum=0.5, lr=args.lr)
    scheduler = MultiStepLR(optimizer, [5, 15, 30], gamma=0.1)
    
    logger.recordparameter()
    model = [backbone, net]
    maxauc = 0
    for epoch in range(epoch_start, epoch_start + args.epoch):
        logger.info("Epoch: {}/{}".format(epoch, epoch_start + args.epoch))
        
        model, losses = train(model, trainloader, device, optimizer)
        with torch.no_grad():
            result = test(model, testloader, device, args, logger)

        logger.recordloss(losses, epoch)
        logger.recordauc(result, epoch)        

        if result.auc() > maxauc:
            maxauc = result.auc()
        if args.savelog:
            logger.savemodel(model, epoch)
        scheduler.step()
    logger.info("best preformance: {:.4f}".format(maxauc))

        

if __name__ == '__main__':
    main()
