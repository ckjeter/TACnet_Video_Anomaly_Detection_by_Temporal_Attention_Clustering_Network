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
import logging
from tqdm import tqdm
import ipdb
from datetime import datetime

from tensorboardX import SummaryWriter 

from src.dataset import FrameFolderDataset
from src.pytorch_i3d import InceptionI3d
from src.backbone import C3D
import src.config as config

@torch.no_grad()
def extract(device, net, input_path):
    dataset = FrameFolderDataset(input_path)
    dataloader = DataLoader(dataset, batch_size = 10, shuffle=False, num_workers=50)
    feature = []
    for data in dataloader:
        frames = data
        frames = frames.to(device)
        output = net(frames)
        feature.append(output.cpu())
    feature = torch.cat(feature, dim=0)
    feature = feature.numpy()
    if config.fix_length:
        feature = np.array_split(feature, config.segment_count)
        length = [len(f) for f in feature]
        feature = [f.mean(axis=0) for f in feature]
        feature = np.array(feature)
        length = np.array(length)
    else:
        length = [1] * feature.shape[0]
        length = np.array(length)
    return feature, length

def path_generate(root, out_path, video):
    if video.find("Normal") >= 0:
        input_path = os.path.join(root, video.split(".")[0], 'frame')
        output_path = os.path.join(out_path, video.split("/")[-1].split("_x")[0] + '.npy')
    else:
        input_path = os.path.join(root, "Anomaly-Videos", video.split(".")[0], 'frame')
        output_path = os.path.join(out_path, video.split("/")[-1].split("_")[0] + '.npy')
        
    return input_path, output_path

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main():
    args = config.parse_args()
    out_root = os.path.join(config.root, 'features', config.token)
    mkdir(out_root)

    trainlist = []
    data_list = np.genfromtxt(os.path.join(config.root, "Anomaly_Train.txt"), dtype=str)
    print("Generating Training List")
    out_path = os.path.join(out_root, 'train')
    mkdir(out_path)
    for video in data_list:
        input_path, output_path = path_generate(config.root, out_path, video)
        trainlist.append((input_path, output_path))

    testlist = []
    data_list_test = np.genfromtxt(
            os.path.join(config.root, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"), dtype=str
    )
    print("Generating Testing List")
    out_path = os.path.join(out_root, 'test')
    mkdir(out_path)
    for data in data_list_test:
        video, category, f1, f2, f3, f4 = data
        if category == 'Normal':
            video = os.path.join("Testing_Normal_Videos_Anomaly", video)
        else:
            video = os.path.join(category, video)
        input_path, output_path = path_generate(config.root, out_path, video)
        testlist.append((input_path, output_path, (int(f1), int(f2), int(f3), int(f4))))

    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Multigpu: {}".format(multi_gpus))

    backbone = C3D()
    backbone.load_state_dict(torch.load("models/c3d.pickle"))

    if multi_gpus:
        backbone = nn.DataParallel(backbone).to(device)
    else:
        backbone = backbone.to(device)
    print("Extracting Train Datas...")
    for paths in trainlist:
        input_path, output_path = paths
        if not os.path.exists(output_path):
            feature, length = extract(device, backbone, input_path)
            output_file = np.array((np.array([]), feature, length), dtype=object)
            np.save(output_path, output_file, allow_pickle=True)
            print("save to {}, shape: {}".format(output_path, feature.shape))
    print("Extracting Test Datas...")
    for paths in testlist:
        input_path, output_path, label = paths
        if not os.path.exists(output_path):
            feature, length = extract(device, backbone, input_path)
            output_file = np.array((np.array(label), feature, length), dtype=object)
            np.save(output_path, output_file, allow_pickle=True)
            print("save to {}, shape: {}".format(output_path, output_file[1].shape))
if __name__ == '__main__':
    main()
