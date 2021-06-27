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
                    X=f, num_clusters=2, distance='cosine', device=self.device, iter_limit=100)
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
            if len(c1[i]) == 0 or len(c2[i]) == 0 or label[i] == 0:
                #normal
                predict = torch.cat((c1[i], c2[i]), dim=0)
                zerolabel = torch.zeros(c1[i].shape[0]+c2[i].shape[0]).to(self.device)
                loss += F.binary_cross_entropy(predict, zerolabel) 
                #loss += abs(maxscore - minscore)
                #return abs(c1.mean() - c2.mean())
            else:
                c_a = c1[i] if c1[i].mean() >= c2[i].mean() else c2[i]
                c_n = c2[i] if c2[i].mean() < c1[i].mean() else c1[i]
                #maxscore = max(c1[i].max(), c2[i].max())
                #minscore = min(c1[i].min(), c2[i].min())
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
                loss += 1 - maxscore + minscore
        return loss

class SmoothLoss(nn.Module):
    def __init__(self, quantize=20.):
        super(SmoothLoss, self).__init__()
        self.quantize = quantize
    def forward(self, output_seg):
        loss = 0
        #for A in output_seg:
        #    loss += (A[0] - A[1])**2
        #    for i in range(1, len(A)-1):
        #        loss += (A[i] - A[i + 1])**2
        for A in output_seg:
            predict_in_range = torch.floor(A.div(1/self.quantize))
            for x in range(0, int(self.quantize)):
                x_count = torch.count_nonzero(predict_in_range == x)
                prob = x_count / len(A)
                if prob != 0:
                    loss += -1 * prob * torch.log2(prob)
        return loss

class SmallLoss(nn.Module):
    def forward(self, output_seg):
        loss = 0
        for A in output_seg:
            loss += A.sum()
        return loss

class MaskLoss(nn.Module):
    def forward(self, attn):
        attn = attn.view(attn.shape[0], -1)
        loss = 1 - torch.max(attn, dim=1)[0] + torch.min(attn, dim=1)[0]
        loss = loss.mean()
        
        return loss
