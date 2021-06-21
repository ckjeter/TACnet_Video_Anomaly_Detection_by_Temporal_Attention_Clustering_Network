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
class UCFCrime_Fast(Dataset):
    def __init__(self, test=False, use_saliency = False):
        self.root = config.root
        self.videolist = []
        self.use_saliency = use_saliency
        self.test = test
        self.weight = 0.3
        if test:
            self.root = os.path.join(self.root, 'test/*')
        else:
            self.root = os.path.join(self.root, 'train/*')
        self.mean = np.load('models/c3d_mean.npy') #N, C, T, H, W
        self.videolist = sorted(glob.glob(self.root))
    def __getitem__(self, index):
        path = self.videolist[index]
        title = os.path.basename(path)
        imgs = sorted(glob.glob(os.path.join(path, 'frame/*')), 
                key = lambda x : int(os.path.basename(x).split('.')[0]))
        imgs = np.array([np.array(Image.open(path), dtype=np.float) for path in imgs])

        if self.use_saliency:
            saliency = sorted(glob.glob(os.path.join(path, 'saliency/*')), 
                    key = lambda x : int(os.path.basename(x).split('.')[0]))
            saliency = np.array([np.array(Image.open(path), dtype=np.float) for path in saliency])
            saliency[saliency > 0] = 1
            saliency[saliency == 0] = self.weight
            saliency = np.repeat(saliency[:, :, :, np.newaxis], 3, axis=3)
            imgs = imgs * saliency
        #os.makedirs(os.path.join(path, 'check'), exist_ok=True)
        #for i in range(512):
        #    img = imgs[i].astype(np.uint8)
        #    cv2.imwrite(os.path.join(path, 'check', '%d.jpg' % i), img)
        imgs = imgs.reshape(32, 16, 112, 112, 3)
        imgs = imgs.transpose(0, 4, 1, 2, 3)
        imgs = imgs - self.mean[0, :, :, 8:120, 30:142]
        imgs = torch.tensor(imgs, dtype=torch.float32)
        
        if self.test:
            label, length = np.load(os.path.join(path, title + '.npy'), allow_pickle=True)
        else:
            length = np.load(os.path.join(path, title + '.npy'), allow_pickle=True)
            if title.find("Normal") >= 0:
                label = 0
            else:
                label = 1
        return title, imgs, label, length
    def __len__(self):
        return len(self.videolist)


class UCFCrime(Dataset):
    def __init__(self, test=False, target = 'frame'):
        self.target = target
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
            #imgs = np.array([np.array(Image.open(path), dtype=np.uint8) for path in clip])
            #imgs = np.array([cv2.imread(path) for path in clip])
            imgs = imgs.transpose(3, 0, 1, 2) # (T, H, W, C) => (C, T, H, W)
            #imgs = (imgs - self.mean[0][:, :imgs.shape[1], :, :])[:, :, 8:120, 30:142]
            imgs = imgs[:, :, 8:120, 30:142]
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
            input_path = os.path.join(root, video.split(".")[0], self.target)
        else:
            input_path = os.path.join(root, "Anomaly-Videos", video.split(".")[0], self.target)
        return input_path
    def __len__(self):
        return len(self.title)




if __name__ == '__main__':
    dataset = UCFCrime()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    for data in dataloader:
        filename, frames, label = data
        ipdb.set_trace()
        print(label)


