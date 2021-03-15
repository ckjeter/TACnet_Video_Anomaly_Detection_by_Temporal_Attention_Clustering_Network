import ipdb
import torch
import torch.nn as nn

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


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
    def forward(self,A):
        loss = (A[1] - A[0])**2
        for i in range(1, len(A)-1):
            loss += (A[i+1] - A[i])**2
        return loss

