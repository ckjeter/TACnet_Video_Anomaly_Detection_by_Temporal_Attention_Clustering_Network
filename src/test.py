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
import cv2
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

from src.backbone import *
from src.util import *
import src.config as config


def bagexpand(bag, length):
    instances = []
    for i, value in enumerate(bag):
        instances += [float(value)] * (length[i])
    return instances

def test(model, loader, device, args, logger):
    backbone, net = model
    net.eval()
    backbone.eval()
    torch.manual_seed(777)
    maxauc = 0
    result = AnomalyResult()
    for i, data in enumerate(loader):
        sys.stdout.write("{}/{}\r".format(i, len(loader)))
        sys.stdout.flush()
        title, imgs, label, length = data
        imgs = imgs.to(device)
        batch, seq_length, channel, clip_length, h, w = imgs.shape
        label = label[0]
        length = length[0]

        if (label == 1).nonzero(as_tuple=True)[0].shape[0] > 0: #anomaly
            baglabel = 1
        else:
            baglabel = 0

        feature = backbone(imgs.squeeze(0)).unsqueeze(0)
        feature, clusters, output_seg, bagoutput, A = net(feature)
        bagoutput = torch.sum(bagoutput, 1)
        result.addbag(bagoutput.view(-1).tolist(), [baglabel])
        
        
        if bagoutput.item() < -1:
            framepredict = [0] * (sum(length).item())
        else:
            segs = output_seg[0].cpu().tolist()
            framepredict = bagexpand(segs, length)
        result.add(title[0], feature, framepredict, label, length)
        if args.p_graph:
            figure = result.predictplot(title[0])
            logger.savefig(figure, os.path.join('performance', title[0] + '.png'))
        if args.c_graph:
            figure = result.clusterplot(title[0])
            logger.savefig(figure, os.path.join('cluster', title[0] + '.png'))
        if args.drawattn:
            A = A.view(-1)
            #A = (A - A.min()) / (A.max() - A.min())
            A = bagexpand(A.cpu().tolist(), length)
            figure = result.predictplot(title[0])
            plt.plot(A, label="Temp Attn", color='green')
            plt.legend()
            logger.savefig(figure, os.path.join('attn', title[0] + '.png'))

    return result 
