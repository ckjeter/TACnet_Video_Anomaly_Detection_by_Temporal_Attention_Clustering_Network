import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from kmeans_pytorch import kmeans
#from .unet_parts import *
import ipdb

class Vis_Attn(nn.Module):
    """ Self attention Layer """""
    def __init__(self,in_dim=9):
        super(Vis_Attn,self).__init__()
        #self.inc = DoubleConv(in_dim, 16)
        #self.down1 = Down(16, 32)
        #self.down2 = Down(32, 64)
        #self.down3 = Down(64, 128)
        #factor = 2
        #self.down4 = Down(128, 256 // factor)
        #self.up1 = Up(256, 128 // factor)
        #self.up2 = Up(128, 64 // factor)
        #self.up3 = Up(64, 32 // factor)
        #self.up4 = Up(32, 16)
        #self.outc = OutConv(16, 1)

        self.D = in_dim // 3
        self.atten_V = nn.Sequential(
                nn.Conv2d(in_channels = in_dim, out_channels = self.D, kernel_size=5, padding=2),
                nn.Tanh()
        )
        self.atten_U = nn.Sequential(
                nn.Conv2d(in_channels = in_dim, out_channels = self.D, kernel_size=5, padding=2),
                nn.Sigmoid()
        )
        self.atten_gate = nn.Conv2d(in_channels = self.D, out_channels = 1, kernel_size=5, padding=2)
        self.attention = nn.Sequential(
                nn.Conv2d(in_channels = in_dim, out_channels = in_dim//3, kernel_size=1),
                nn.Tanh(),
                nn.Conv2d(in_channels = in_dim//3, out_channels=1, kernel_size=1)
                )

    def forward(self,x):
        """
        inputs :
        x : input feature maps( B X C X W X H )
        returns :
        out : self attention value + input feature 
        attention: B X N X N (N is Width*Height)
        """
        batch, channel, w, h = x.shape
        input = x
        #x1 = self.inc(x)
        #x2 = self.down1(x1)
        #x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        #x = self.up2(x, x3)
        #x = self.up3(x, x2)
        #x = self.up4(x, x1)
        #attention = self.outc(x)
        #attention = torch.relu(attention)
        #attention = attention / attention.max()

        x_v = self.atten_V(x)
        x_u = self.atten_U(x)
        attention = self.atten_gate(torch.mul(x_v, x_u)).view(batch, 1, -1)
        attention = torch.sigmoid(attention).view(batch, 1, w, h)
        
        #attention = torch.softmax(attention, dim=2).view(batch, 1, w, h)
        #a_max = attention.max(dim=2, keepdim=True)[0]
        #a_min = attention.min(dim=2, keepdim=True)[0]
        #attn_norm = (attention - a_min) / (a_max - a_min)
        #attn_norm = attn_norm.view(batch, 1, w, h)

        #out = torch.mul(input[:, 3:6, :, :], attention) + input[:, 3:6, :, :]
        out = torch.mul(input[:, 3:6, :, :], attention) 
        return out, attention

class Temp_Attn(nn.Module):
    def __init__(self, args, device):
        super(Temp_Attn, self).__init__()
        self.attention_type = args.attention_type
        self.device = device
        self.L = 256
        self.D = 256
        self.mlp1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.L)
        )
        self.rnn = nn.GRU(4096, self.L, 2, batch_first=True)
        self.tsn = nn.Conv1d(in_channels=4096, out_channels=self.L, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_gate = nn.Linear(self.D, 1)
        self.c_bag = self.classifier(self.L)
        self.c_segment = self.classifier(self.L + 1)
    def classifier(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.D),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.D, int(self.D / 2)),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(int(self.D / 2), 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def maxminnorm(self, A):
        A = A.view(-1)
        return (A - min(A)) / (max(A)-min(A))

    def forward(self, input):
        feature = torch.nan_to_num(input) #1, 32, self.L
        batch_size = input.shape[0]
        #key point
        #feature = self.mlp1(feature)
        #feature, hidden = self.rnn(feature)
        feature = self.tsn(feature.transpose(1, 2)).transpose(1, 2)

        #Attention path
        if self.attention_type == 'normal':
            A = self.attention(feature) 
        elif self.attention_type == 'gate':
            A_V = self.attention_V(feature)
            A_U = self.attention_U(feature)
            A = self.attention_gate(A_V * A_U)
        #A: 1, 32, 1
        A = torch.transpose(A, 1, 2) #1, 1, 32
        bag = torch.bmm(F.softmax(A, dim=2), feature).squeeze(1) #1, self.L
        #bag = torch.bmm(A, feature)
        #output1 = torch.sigmoid(self.c_bag(torch.cat((bag, hidden[-1]), dim=1))).view(-1)
        output1 = torch.sigmoid(self.c_bag(bag)).view(-1) #1, 1

        #Cluster path
        #A = 2 * torch.sigmoid(A)
        #A = A.view(1, -1, 1).expand(-1, -1, feature.shape[-1])
        #feature = feature * A
        #A = self.c_segment(feature).squeeze(2)

        #key point
        #concate? multiply?
        output_seg = self.c_segment(torch.cat((feature, A.view(batch_size, 32, 1)), dim=2)).squeeze(2)
        output_seg = torch.sigmoid(output_seg)
        #1, 32
        #attention_boost = 2 * torch.sigmoid(A).squeeze(1)
        #output_seg = torch.sigmoid(output_seg * attention_boost)
        #dont del

        #output_seg = torch.sigmoid(A)
        #A = torch.transpose(A.unsqueeze(2), 1, 2)
        #bag = torch.bmm(F.softmax(A, dim=2), feature).squeeze(1)
        #output1 = torch.sigmoid(self.c_bag(bag)).view(-1)
        cluster1 = []
        cluster2 = []
        output2 = torch.tensor([]).to(self.device)
        for i in range(feature.shape[0]):
            #f = feature_dirty[i][~torch.any(feature_dirty[i].isnan(),dim=1)]
            f = feature[i]
            cluster, centers = kmeans(
                X=f, num_clusters=2, distance='cosine', device=self.device, iter_limit=100)
            #output_seg = torch.sigmoid(output_seg.view(-1))
            c1 = torch.nonzero(cluster==0).view(-1).to(self.device)
            c2 = torch.nonzero(cluster!=0).view(-1).to(self.device)
            c1 = torch.index_select(output_seg[i], 0, c1)
            c2 = torch.index_select(output_seg[i], 0, c2)

            #key point
            out = max(c1.mean(), c2.mean()).view(-1)
            #out = output_seg[i].max().view(-1)

            cluster1.append(c1)
            cluster2.append(c2)
            output2 = torch.cat((output2, out), dim=0)
        weight = 0.5
        output1 = output1 * weight
        output2 = output2 * (1 - weight)
        output = torch.cat((output1.view(-1, 1), output2.view(-1, 1)), 1)
        #output = output2
        return feature, (cluster1, cluster2), output_seg, output, A

class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        '''
        h = self.dropout(h)
        h = self.relu(self.fc7(h))
        h = self.dropout(h)

        logits = self.fc8(h)
        probs = self.softmax(logits)

        return probs
        '''
        return h
