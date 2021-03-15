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
import src.config as config
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
            if video.find("Normal") >= 0:
                path = os.path.join(self.root, video.split(".")[0], 'frame')
            else:
                path = os.path.join(self.root, "Anomaly-Videos", video.split(".")[0], 'frame')
            framelist = sorted(glob.glob(path + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
            self.frames.append(framelist)
            if video.find("Normal") >= 0:
                self.labels.append(0)
            else:
                self.labels.append(1)
    def __getitem__(self, index):
        print(self.videos[index], len(self.frames[index]))
        framelist = self.frames[index]
        label = self.labels[index]
        frames = [Image.open(frame).convert("RGB") for frame in framelist]
        frames = torch.tensor([self.transforms(frame).numpy() for frame in frames])
        frames = list(torch.split(frames, config.segment_length, dim=0))
        frames[-1] = F.pad(frames[-1], pad=(0,0,0,0,0,0,0,config.segment_length-frames[-1].shape[0]))
        frames = torch.stack(frames)
        return self.videos[index], frames, label

    def __len__(self):
        return len(self.frames)

class FrameFolderDataset(Dataset):
    def __init__(self, folder_dir, clip_size=config.segment_length):
        self.framelist = sorted(glob.glob(folder_dir + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
        self.clips = list(self.chunks(self.framelist, clip_size))
        self.mean = np.load('models/c3d_mean.npy') #N, C, T, H, W
        
    def __getitem__(self, index):
        img_paths = self.clips[index]
        imgs = np.array([np.array(Image.open(path).resize((171, 128), Image.BICUBIC), dtype=np.float) for path in img_paths])
        imgs = imgs.transpose(3, 0, 1, 2) # (T, H, W, C) => (C, T, H, W)
        if imgs.shape[1] < config.segment_length:
            imgs = np.pad(imgs, ((0, 0), (0, config.segment_length - imgs.shape[1]), (0, 0), (0, 0)), 'constant')
        imgs = (imgs - self.mean[0])[:, :, 8:120, 30:142]
        # imgs = imgs[:, :, 8:120, 30:142]
        imgs = torch.tensor(imgs, dtype=torch.float32)
        return imgs

    def chunks(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def __len__(self):
        return len(self.clips)

class SegmentDataset(Dataset):
    def __init__(self, folder_dir, test=False):
        self.test = test
        self.labels = []
        self.features = glob.glob(folder_dir + '/*')
        self.titles = [os.path.basename(path).split(".")[0] for path in self.features]

    def __getitem__(self, index):
        title = self.titles[index]
        if title.find("Normal") >= 0:
            label = 0
        else:
            label = 1
        feature = np.load(self.features[index], allow_pickle=True)
        if self.test:
            return title, feature[1], feature[0]
        return title, feature, label
    def __len__(self):
        return len(self.titles)





if __name__ == '__main__':
    dataset = SegmentDataset(folder_dir='features/test', test=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for data in dataloader:
        filename, frames, label = data
        ipdb.set_trace()
        print(label)


