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
    backbone, net, atten = model
    c3d_mean = torch.FloatTensor(loader.dataset.mean[0]).to(device)
    net.eval()
    backbone.eval()
    atten.eval()
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

        imgs_seq = imgs.transpose(2, 3).squeeze(0)
        imgs_seq = imgs_seq.reshape(-1, channel, h, w)
        first = imgs_seq[0].unsqueeze(0)
        last = imgs_seq[-1].unsqueeze(0)
        imgs_prev = torch.cat((first, imgs_seq[:-1]), dim=0)
        imgs_post = torch.cat((imgs_seq[1:], last), dim=0)
        imgs_seq = torch.cat((imgs_prev, imgs_seq, imgs_post), dim=1)

        imgs_attn, attn = atten(imgs_seq)
        imgs = imgs_attn.view(batch, seq_length, clip_length, channel, h, w).transpose(2, 3)
        imgs = imgs[:, :] - c3d_mean[:, :, 8:120, 30:142]

        if label[0] < 0:
            baglabel = 0
        else:
            baglabel = 1
        feature = backbone(imgs.squeeze(0)).unsqueeze(0)
        feature, clusters, output_seg, bagoutput, A = net(feature)
        bagoutput = torch.sum(bagoutput, 1)
        result.addbag(bagoutput.view(-1).tolist(), [baglabel])
        
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
        if args.drawattn:
            A = A.view(-1)
            #A = (A - A.min()) / (A.max() - A.min())
            A = bagexpand(A.cpu().tolist(), length)
            figure = result.predictplot(title[0])
            plt.plot(A, label="Temp Attn", color='green')
            plt.legend()
            logger.savefig(figure, os.path.join('attn', title[0] + '.png'))

        if args.drawmask:
            for count in range(0, 512):
                seg_count = count // 16
                seg_pos = count % 16
                step = length[seg_count] // 16
                realcount = length[:seg_count].sum() + (step * seg_pos)
                realcount = realcount.item()
                mask = attn[count].view(h, w, 1).cpu().numpy()
                #mask = (mask - mask.min()) / (mask.max() - mask.min())
                #mask = ((1 - mask) * 256).astype(np.uint8)
                mask = (mask * 256).astype(np.uint8)
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
                logger.savefig(mask, os.path.join('mask', title[0], str(realcount)+'.png'))
    return result 
