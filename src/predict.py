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
from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle

from tensorboardX import SummaryWriter 

from src.dataset import FrameFolderDataset, SegmentDataset
from src.pytorch_i3d import InceptionI3d
from src.backbone import C3D, Attention
from src.loss import ClusterLoss, SmoothLoss
from src.util import Averager, Scorer, AnomalyScorer
import src.util as util
import src.config as config


def bagexpand(bag, K=16):
    instances = []
    for value in bag:
        instances += [float(value)] * 16
    return instances

def instanceplot(title, predict, label, acc):
    figure, ax = plt.subplots()
    plt.plot(predict)
    plt.title(title)
    plt.xlabel('Frame number')
    plt.ylabel('Anomaly score')
    for i in range(0, len(label), 2):
        if label[i] != -1:
            ax.add_patch(Rectangle((label[i], 0), label[i+1]-label[i], 1, color='red', alpha=0.5))
    try:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        auc = acc.auc()
        text = "AUC: " + str(auc)[:4]
        plt.text(0, 1.1, text, bbox=props)
    except:
        pass
    return figure
def clusterplot(title, feature, label):
    figure, ax = plt.subplots()
    downsample = TSNE(n_components=2).fit_transform(feature.squeeze(0).cpu().detach())
    newlabel = [0] * feature.shape[1]
    for i in range(0, 4, 2):
        if label[i] != -1:
            newlabel[int(label[i]/16):int(label[i+1]/16)] = [1] * (int(label[i+1]/16)-int(label[i]/16))
    newlabel = np.array(newlabel)
    plt.title(title[0])
    plt.scatter(x=downsample.T[0], y=downsample.T[1], c=newlabel)
    return figure


def predict(net, loader, device, args):
    maxauc = 0
    result = AnomalyScorer()
    for data in loader:
        singleacc = Scorer()
        title, feature, label = data
        feature = feature.to(device)
        label = label.squeeze(0).to(device)

        if label[0] < 0:
            baglabel = 0
        else:
            baglabel = 1
        feature, A, bagoutput = net(feature)

            
        result.addbag(bagoutput.view(-1).tolist(), [baglabel])
        #test my idea
        #atten_weight = A
        #A = net.classification(feature).view(-1)
        #A = net.maxminnorm(A * atten_weight)
        
        if bagoutput.item() < 0.1:
            framepredict = [0] * (len(A) * 16)
        else:
            framepredict = bagexpand(A)
        
        framelabel = [0] * len(framepredict)
        
        for i in range(0, 4, 2):
            if label[i] != -1:
                framelabel[label[i]:label[i+1]] = [1] * (label[i+1]-label[i])
        if args.tsne and title[0] == 'Arson011':
            figure = clusterplot(title[0], feature, label)
            figure.savefig(title[0] + "_features.png")

        result.add(title[0], framepredict, framelabel)
        singleacc.add(framepredict, framelabel)
        if args.draw:
            figure = instanceplot(title[0], framepredict, label.tolist(), singleacc)
            figure.savefig(os.path.join('image', title[0]+'.png'))
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
    net = Attention(args.attention_type)
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
    logger.auc_types(result)
    print(result.auc())

