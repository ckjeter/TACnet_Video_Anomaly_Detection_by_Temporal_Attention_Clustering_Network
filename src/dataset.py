import torch
import cv2
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
import ipdb
import math
import glob
import numpy as np
import src.config as config
from tqdm import tqdm
class ShanghaiTech(Dataset):
    def __init__(self, mode='train', use_saliency = 1, check = False, image_path = ''):
        self.root = config.root
        self.videolist = []
        self.mode = mode
        self.check = check
        self.image_path = image_path
        self.root = os.path.join(self.root, mode, '*')
        self.videolist = sorted(glob.glob(self.root))
    def __getitem__(self, index):
        path = self.videolist[index]
        title = os.path.basename(path)
        imgs, label, length = np.load(path, allow_pickle=True)

        if self.check:
            if self.image_path.find(".pth") >= 0:
                name = os.path.basename(os.path.dirname(self.image_path))
            else:
                name = self.image_path
            target = os.path.join(config.root, 'image', name, 'check', title)
            os.makedirs(target, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(target + '.mp4', fourcc, 30, (112*3,112))
            for i in range(32):
                for j in range(16):
                    img = imgs[i][j].astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    out.write(frame)
                    cv2.imwrite(os.path.join(target, '%d.jpg' % (16 * i + j)), img)
            out.release()
        imgs = imgs.transpose(0, 4, 1, 2, 3)

        selfmean = imgs.mean(axis=0)
        imgs = imgs - selfmean
        imgs = torch.tensor(imgs, dtype=torch.float32)
        
        return title, imgs, label, length
    def __len__(self):
        return len(self.videolist)

if __name__ == '__main__':
    dataset = UCFCrime()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for data in dataloader:
        filename, frames, label = data
        ipdb.set_trace()
        print(label)


