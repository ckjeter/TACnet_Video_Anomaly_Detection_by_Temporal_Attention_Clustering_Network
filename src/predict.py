import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

import csv
import random
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import ipdb
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tensorboardX import SummaryWriter 

from src.dataset import FrameFolderDataset, SegmentDataset
from src.pytorch_i3d import InceptionI3d
from src.backbone import C3D, Attention
from src.loss import ClusterLoss, SmoothLoss
from src.util import Averager, Scorer, AnomalyResult
import src.util as util
import src.config as config


def bagexpand(bag, K=16):
    instances = []
    for value in bag:
        instances += [float(value)] * 16
    return instances

def predict(net, loader, device, args):
    net.eval()
    torch.manual_seed(777)
    maxauc = 0
    result = AnomalyResult()
    for data in loader:
        title, feature, label = data
        feature = feature.to(device)
        label = label.squeeze(0).to(device)

        if label[0] < 0:
            baglabel = 0
        else:
            baglabel = 1
        feature, clusters, output_seg, bagoutput = net(feature)
            
        result.addbag(bagoutput.view(-1).tolist(), [baglabel])
        #test my idea
        #atten_weight = A
        #A = net.classification(feature).view(-1)
        #A = net.maxminnorm(A * atten_weight)
        if bagoutput.item() < 0.5:
            framepredict = [0] * (len(output_seg) * 16)
        else:
            framepredict = bagexpand(output_seg)
        result.add(title[0], feature, framepredict, label)
        if args.draw:
            figure = result.predictplot(title[0])
            figure.savefig(os.path.join('image', 'performance', title[0]+'.png'))
            plt.close(figure)
            figure = result.clusterplot(title[0])
            figure.savefig(os.path.join('image', 'cluster', title[0]+'.png'))
            plt.close(figure)
    return result 

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
    result = predict(net, testloader, device, args)
    roc = result.roccurve()
    roc.savefig("ROC.png")
    logger.auc_types(result)
    print(result.auc())
