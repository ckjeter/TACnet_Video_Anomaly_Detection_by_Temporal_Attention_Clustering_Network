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
        feature = feature.squeeze(0)
        if label == 1:
            _, centers = kmeans(
                X=feature, num_clusters=2, distance='euclidean', device=self.device, tqdm_flag=False)
            centers = centers.to(self.device)
            dst = torch.cdist(centers[0].view(1, 1, -1), centers[1].view(1, 1, -1))
            return 1 / dst.view(-1)
        else:
            center = torch.mean(feature, dim=0)
            dst = torch.cdist(feature.unsqueeze(0), center.view(1, 1, -1))
            dst = torch.mean(dst.view(-1))
            dst = torch.minimum(dst, torch.tensor(10).to(self.device))
            return dst.view(-1)

class InnerBagLoss(nn.Module):
    def __init__(self, device):
        super(InnerBagLoss, self).__init__()
        self.device = device
    def forward(self, clusters, label):
        c1, c2 = clusters
        c_a = c1 if c1.mean() >= c2.mean() else c2
        c_n = c2 if c2.mean() < c1.mean() else c1
        maxscore = max(c1.max(), c2.max())
        minscore = min(c1.min(), c2.min())
        if label == 0:
            #normal
            predict = torch.cat((c1, c2), dim=0)
            zerolabel = torch.zeros(c1.shape[0]+c2.shape[0]).to(self.device)
            loss = F.binary_cross_entropy(predict, zerolabel) 
            return loss
            #return abs(c1.mean() - c2.mean())
            #return abs(maxscore - minscore)
        else:
            #anomaly
            onelabel = torch.ones(c_a.shape[0]).to(self.device)
            zerolabel = torch.zeros(c_n.shape[0]).to(self.device)
            loss = F.binary_cross_entropy(torch.cat((c_a, c_n), 0), torch.cat((onelabel, zerolabel), 0))
            return loss
            #return max(0, 1 - maxscore + minscore)
            #return max(0, 1 - abs(c1.mean() - c2.mean()))
class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
    def forward(self,A):
        loss = (A[0] - A[1])**2
        for i in range(1, len(A)-1):
            loss += (A[i] - A[i + 1])**2
        return loss

class SmallLoss(nn.Module):
    def __init__(self):
        super(SmallLoss, self).__init__()
    def forward(self, A):
        return A.sum()


