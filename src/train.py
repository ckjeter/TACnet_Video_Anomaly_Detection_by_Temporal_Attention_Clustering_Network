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

from tensorboardX import SummaryWriter 

from src.dataset import FrameFolderDataset, SegmentDataset
from src.pytorch_i3d import InceptionI3d
from src.backbone import C3D, Attention
from src.loss import ClusterLoss, SmoothLoss
from src.util import Averager
import src.config as config
        
def train(net, trainloader, device, optimizer):
    bagLoss = nn.BCELoss().to(device)
    clusterLoss = ClusterLoss(device).to(device)
    smoothLoss = SmoothLoss().to(device)
    bag_loss_count = Averager()
    cluster_minloss_count = Averager()
    cluster_maxloss_count = Averager()
    smooth_loss_count = Averager()

    for i, data in enumerate(trainloader):
        sys.stdout.write("    Train Batch: {}/{}\r".format(i, len(trainloader)))
        sys.stdout.flush()
        title, feature, label = data
        feature = feature.to(device)
        label = label.to(device)
        feature, A, output = net(feature)
        
        bag_loss = bagLoss(output.view(-1), label.view(-1).type(torch.float))
        cluster_loss = clusterLoss(feature, label[0])
        smooth_loss = smoothLoss(A)

        bag_loss_count.add(bag_loss.item())
        smooth_loss_count.add(smooth_loss.item())
        if label[0] == 0:
            cluster_minloss_count.add(cluster_loss.item())
        else:
            cluster_maxloss_count.add(cluster_loss.item())

        #loss = 1 * bag_loss + 1 * cluster_loss + 1 * smooth_loss
        loss = bag_loss + smooth_loss
        #loss = bag_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses = [bag_loss_count.item(), cluster_maxloss_count.item(), 
                cluster_minloss_count.item(), smooth_loss_count.item()]
    return net, losses

if __name__ == '__main__':
    pass
