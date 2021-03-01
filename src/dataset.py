import torch
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
import ipdb
import glob
import numpy as np
import src.config as config
class UCFCrime(Dataset):
    def __init__(self, mode='train'):
        self.root = config.root_UCFCrime
        self.videolist = []
        self.mode = mode
        self.root = os.path.join(self.root, mode, '*')
        self.videolist = sorted(glob.glob(self.root))
    def __getitem__(self, index):
        path = self.videolist[index]
        title = os.path.basename(path)
        imgs, label, length = np.load(os.path.join(path, title + '.npy'), allow_pickle=True)
        if self.mode.find("Normal") >= 0:
            if title.find("Normal") >= 0:
                label = 0
            else:
                label = 1
        imgs = imgs.transpose(0, 4, 1, 2, 3)

        selfmean = imgs.mean(axis=0)
        imgs = imgs - selfmean
        imgs = torch.tensor(imgs, dtype=torch.float32)
        
        return title, imgs, label, length
    def __len__(self):
        return len(self.videolist)
class ShanghaiTech(Dataset):
    def __init__(self, mode='train'):
        self.root = config.root_SH
        self.videolist = []
        self.mode = mode
        self.root = os.path.join(self.root, mode, '*')
        self.videolist = sorted(glob.glob(self.root))
    def __getitem__(self, index):
        path = self.videolist[index]
        title = os.path.basename(path)
        imgs, label, length = np.load(path, allow_pickle=True)

        imgs = imgs.transpose(0, 4, 1, 2, 3)

        selfmean = imgs.mean(axis=0)
        imgs = imgs - selfmean
        imgs = torch.tensor(imgs, dtype=torch.float32)

        return title, imgs, label, length
    def __len__(self):
        return len(self.videolist)

