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


def bagexpand(bag, length):
    instances = []
    for i, value in enumerate(bag):
        instances += [float(value)] * (length[i])
    return instances

def test(model, loader, device, args, logger):
    backbone, net = model
    net.eval()
    torch.manual_seed(777)
    maxauc = 0
    result = AnomalyResult()
    for i, data in enumerate(loader):
        sys.stdout.write("{}/{}\r".format(i, len(loader)))
        sys.stdout.flush()
        title, imgs, label, length = data
        imgs = imgs.to(device)
        label = label[0]
        length = length[0]

        if label[0] < 0:
            baglabel = 0
        else:
            baglabel = 1
        feature = backbone(imgs.squeeze(0)).unsqueeze(0)
        feature, clusters, output_seg, bagoutput, A = net(feature)
        bagoutput = torch.sum(bagoutput, 1)
        result.addbag(bagoutput.view(-1).tolist(), [baglabel])
        
        #test my idea
        #atten_weight = A
        #A = net.classification(feature).view(-1)
        #A = net.maxminnorm(A * atten_weight)
        if bagoutput.item() < -1:
            framepredict = [0] * (sum(length).item())
        else:
            framepredict = bagexpand(output_seg[0].cpu().tolist(), length)
        result.add(title[0], feature, framepredict, label, length)
        if args.p_graph:
            figure = result.predictplot(title[0])
            logger.savefig(figure, os.path.join('performance', title[0] + '.png'))
        if args.c_graph:
            figure = result.clusterplot(title[0])
            logger.savefig(figure, os.path.join('cluster', title[0] + '.png'))
    return result 
