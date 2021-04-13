import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from kmeans_pytorch import kmeans
from scipy.spatial import distance
import math

class ClusterLoss(nn.Module):
    def __init__(self, device):
        super(ClusterLoss, self).__init__()
        self.device = device
    def forward(self, feature, label):
        loss = 0
        for i in range(feature.shape[0]):
            #f = feature[i][~torch.any(feature[i].isnan(),dim=1)]
            f = feature[i]
            if label[i] == 1:
                _, centers = kmeans(
                    X=f, num_clusters=2, distance='euclidean', device=self.device)
                centers = centers.to(self.device)
                dst = torch.cdist(centers[0].view(1, 1, -1), centers[1].view(1, 1, -1))
                loss += 1 / dst.view(-1)
            else:
                center = torch.mean(f, dim=0)
                dst = torch.cdist(f.unsqueeze(0), center.view(1, 1, -1))
                dst = torch.mean(dst.view(-1))
                dst = torch.minimum(dst, torch.tensor(10).to(self.device))
                loss += dst.view(-1)
        return loss

class InnerBagLoss(nn.Module):
    def __init__(self, device):
        super(InnerBagLoss, self).__init__()
        self.device = device
    def forward(self, clusters, label):
        c1, c2 = clusters
        loss = 0
        for i in range(label.shape[0]):
            c_a = c1[i] if c1[i].max() >= c2[i].max() else c2[i]
            c_n = c2[i] if c2[i].max() < c1[i].max() else c1[i]
            maxscore = max(c1[i].max(), c2[i].max())
            minscore = min(c1[i].min(), c2[i].min())
            if label[i] == 0:
                #normal
                predict = torch.cat((c1[i], c2[i]), dim=0)
                zerolabel = torch.zeros(c1[i].shape[0]+c2[i].shape[0]).to(self.device)
                loss += F.binary_cross_entropy(predict, zerolabel) 
                #loss += abs(maxscore - minscore)
                #return abs(c1.mean() - c2.mean())
            else:
                #anomaly
                onelabel = torch.ones(c_a.shape[0]).to(self.device)
                #zerolabel = torch.zeros(c_n.shape[0]).to(self.device)
                zerolabel = torch.zeros(c_n.shape[0]).to(self.device)
                loss += F.binary_cross_entropy(torch.cat((c_a, c_n), 0), torch.cat((onelabel, zerolabel), 0))
                #loss += max(0, 1 - maxscore + minscore)
                #return max(0, 1 - maxscore + minscore)
                #return max(0, 1 - abs(c1.mean() - c2.mean()))
        return loss

class MaxminLoss(nn.Module):
    def forward(self, output_seg, label):
        loss = 0
        for i in range(label.shape[0]):
            maxscore = output_seg[i].max()
            minscore = output_seg[i].min()
            if label[i] == 0:
                loss += abs(maxscore - minscore)
            else:
                loss += max(0, 1 - maxscore + minscore)
        return loss

class SmoothLoss(nn.Module):
    def forward(self, output_seg):
        loss = 0
        for A in output_seg:
            loss += (A[0] - A[1])**2
            for i in range(1, len(A)-1):
                loss += (A[i] - A[i + 1])**2
        return loss

class SmallLoss(nn.Module):
    def forward(self, output_seg):
        loss = 0
        for A in output_seg:
            loss += A.sum()
        return loss


