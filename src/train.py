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

from src.backbone import *
from src.loss import *
from src.util import Averager
import src.config as config
        
def train(model, trainloader, device, optimizer):
    backbone, net = model
    backbone.train()
    net.train()
    bagLoss = nn.BCELoss().to(device)
    clusterLoss = ClusterLoss(device).to(device)
    smoothLoss = SmoothLoss().to(device)
    innerbagLoss = InnerBagLoss(device).to(device)
    maxminLoss = MaxminLoss().to(device)
    smallLoss = SmallLoss().to(device)

    bag_loss_count = Averager()
    cluster_minloss_count = Averager()
    cluster_maxloss_count = Averager()
    cluster_loss_count = Averager()
    innerloss_normal_count = Averager()
    innerloss_anomaly_count = Averager()
    maxminloss_count = Averager()
    innerloss_count = Averager()
    smooth_loss_count = Averager()
    smallloss_count = Averager()

    for i, data in enumerate(trainloader):
        sys.stdout.write("    Train Batch: {}/{}\r".format(i, len(trainloader)))
        sys.stdout.flush()
        title, imgs, label, length = data
        imgs = imgs.to(device)
        batch, seq_length, channel, clip_length, h, w = imgs.shape
        label = label.to(device)


        feature = backbone(imgs.view(-1, 3, 16, 112, 112)).view(batch, 32, -1)
        feature, clusters, output_seg, output_bag, A = net(feature)
        output = torch.sum(output_bag, 1)

        bag_loss = bagLoss(output.view(-1), label.view(-1).type(torch.float))
        cluster_loss = clusterLoss(feature, label)
        innerbag_loss = innerbagLoss(clusters, label)
        smooth_loss = smoothLoss(output_seg)
        small_loss = smallLoss(output_seg)
        maxmin_loss = maxminLoss(output_seg, label)

        bag_loss_count.add(bag_loss.item())
        smooth_loss_count.add(smooth_loss.item())
        cluster_loss_count.add(cluster_loss.item())
        innerloss_count.add(innerbag_loss.item())
        maxminloss_count.add(maxmin_loss.item())
        smallloss_count.add(small_loss.item())

        parameter = config.loss_parameter
        losses = [bag_loss, cluster_loss, innerbag_loss, maxmin_loss, smooth_loss, small_loss]
        loss = sum([p * l for p, l in zip(parameter, losses)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses = [bag_loss_count.item(), cluster_loss_count.item(), smooth_loss_count.item(),
                maxminloss_count.item(), innerloss_count.item(), smallloss_count.item()]
    return [backbone, net], losses

if __name__ == '__main__':
    pass
