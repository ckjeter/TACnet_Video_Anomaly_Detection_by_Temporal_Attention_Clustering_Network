import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os
import ipdb
import glob
import numpy as np
import config
from tqdm import tqdm

class UCFCrime(Dataset):
    def __init__(self):
        self.root = config.root           
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.videos = []
        self.frames = []
        self.labels = []
        data_list = np.genfromtxt(os.path.join(self.root, "Anomaly_Train.txt"), dtype=str)
        print("dataset init...")
        for video in tqdm(data_list):
            self.videos.append(video)
            path = os.path.join(self.root, video.split(".")[0], 'frame')
            framelist = sorted(glob.glob(path + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
            self.frames.append(framelist)
            if video.find("Normal"):
                self.labels.append(0)
            else:
                self.labels.append(1)
    def __getitem__(self, index):
        print(self.videos[index], len(self.frames[index]))
        framelist = self.frames[index]
        label = self.labels[index]
        frames = [Image.open(frame).convert("RGB") for frame in framelist]
        frames = torch.tensor([self.transforms(frame).numpy() for frame in frames])
        frames = torch.split(frames, config.segment_length, dim=0)
        frames[-1] = F.pad(frames[-1], (0,0,0,0,0,0,0,config.segment_length-frames[-1].shape[0]))
        frames = torch.stack(frames)
        return frames, label

    def __len__(self):
        return len(self.frames)
       
if __name__ == '__main__':
    dataset = UCFCrime()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for data in dataloader:
        frames, label = data
        print(label)
        ipdb.set_trace()


