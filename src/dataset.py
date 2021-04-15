import torch
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

class UCFCrime(Dataset):
    def __init__(self, test=False):
        self.test = test
        self.root = config.root           
        self.video = []
        self.label = []
        self.title = []
        if not test:
            datalist = np.genfromtxt(
                    os.path.join(config.root, "Anomaly_Train.txt"), dtype=str)
            for video in datalist:
                input_path = self.path_generate(config.root, video)
                self.title.append(self.gettitle(video))
                if input_path.find("Normal") >= 0:
                    self.label.append(0)
                else:
                    self.label.append(1)
                self.video.append(input_path)
        else:
            datalist = np.genfromtxt(
                    os.path.join(
                        config.root, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
                , dtype=str)
            for data in datalist:
                video, category, f1, f2, f3, f4 = data
                self.title.append(self.gettitle(video))
                if category == 'Normal':
                    video = os.path.join("Testing_Normal_Videos_Anomaly", video)
                else:
                    video = os.path.join(category, video)
                input_path = self.path_generate(config.root, video)
                self.video.append(input_path)
                self.label.append(np.array([int(f1), int(f2), int(f3), int(f4)]))
        self.mean = np.load('models/c3d_mean.npy') #N, C, T, H, W
    def __getitem__(self, index):
        title = self.title[index]
        label = self.label[index]
        folder_dir = self.video[index]
        clips, length = self.clip(folder_dir)
        return title, clips, label, length
    def clip(self, folder_dir):
        framelist = sorted(glob.glob(folder_dir + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
        framelist = np.array(framelist)
        clips = np.array_split(framelist, config.segment_count)
        sliceclips = []
        length = [len(c) for c in clips]
        for c in clips:
            try:
                clip = [cc[0] for cc in np.array_split(c, 16)]
            except:
                clip = c
            imgs = np.array([np.array(Image.open(path).resize((171, 128), Image.BICUBIC), dtype=np.float) for path in clip])
            imgs = imgs.transpose(3, 0, 1, 2) # (T, H, W, C) => (C, T, H, W)
            #imgs = imgs[:, :, 8:120, 30:142]
            imgs = (imgs - self.mean[0][:, :imgs.shape[1], :, :])[:, :, 8:120, 30:142]
            if imgs.shape[1] < config.segment_length:
                imgs = np.pad(imgs, ((0, 0), (0, config.segment_length - imgs.shape[1]), (0, 0), (0, 0)), 'constant')
            imgs = torch.tensor(imgs, dtype=torch.float32)
            sliceclips.append(imgs)
        sliceclips = np.stack(sliceclips, axis=0)
        return sliceclips, np.array(length)
    def gettitle(self, video):
        video = os.path.basename(video)
        if video.find("Normal") >= 0:
            title = video.split("_x")[0]
        else:
            title = video.split("_")[0]
        return title
    def path_generate(self, root, video):
        if video.find("Normal") >= 0:
            input_path = os.path.join(root, video.split(".")[0], 'frame')
        else:
            input_path = os.path.join(root, "Anomaly-Videos", video.split(".")[0], 'frame')
        return input_path
    def __len__(self):
        return len(self.title)

class FrameFolderDataset(Dataset):
    def __init__(self, folder_dir, clip_size=config.segment_length):
        self.framelist = sorted(glob.glob(folder_dir + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
        self.clips = list(self.chunk_by_clip(self.framelist, clip_size))
        self.clip_size = clip_size
        self.mean = np.load('models/c3d_mean.npy') #N, C, T, H, W
    def __getitem__(self, index):
        img_paths = self.clips[index]
        imgs = np.array([np.array(Image.open(path).resize((171, 128), Image.BICUBIC), dtype=np.float) for path in img_paths])
        imgs = imgs.transpose(3, 0, 1, 2) # (T, H, W, C) => (C, T, H, W)
        # imgs = imgs[:, :, 8:120, 30:142]
        imgs = (imgs - self.mean[0][:, :imgs.shape[1], :, :])[:, :, 8:120, 30:142]
        if imgs.shape[1] < self.clip_size:
            imgs = np.pad(imgs, ((0, 0), (0, self.clip_size - imgs.shape[1]), (0, 0), (0, 0)), 'constant')
        imgs = torch.tensor(imgs, dtype=torch.float32)
        return imgs

    def chunk_by_clip(self, lst, n):
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
        data = np.load(self.features[index], allow_pickle=True)
        framelabel = data[0]
        feature = data[1]
        length = data[2]
        if self.test:
            return title, feature, framelabel, length
        return title, feature, label, length
    def __len__(self):
        return len(self.titles)



if __name__ == '__main__':
    dataset = UCFCrime()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for data in dataloader:
        filename, frames, label = data
        ipdb.set_trace()
        print(label)


