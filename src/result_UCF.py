import torch
import cv2
import csv
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter 
import logging
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, auc
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import src.config as config 
import glob
import os
import ipdb

class Scorer():
    def __init__(self):
        self.predict = []
        self.label = []
    def add(self, predict, label):
        self.predict += predict
        self.label += label
    def auc(self):
        return roc_auc_score(self.label, self.predict)

class AnomalyType(Scorer):
    def __init__(self, category):
        super().__init__()
        self.category = category
        self.count = 0
    def add(self, predict, label):
        super().add(predict, label)
        self.count += 1

class AnomalyVideo(Scorer):
    def __init__(self, title, feature, predict, label, rawlabel, length):
        super().__init__()
        self.title = title
        self.feature = feature
        self.predict = predict
        self.label = label
        self.rawlabel = rawlabel
        self.length = length
    def clusterplot(self):
        figure, ax = plt.subplots()
        downsample = TSNE(n_components=2).fit_transform(self.feature.squeeze(0).cpu().detach())
        framelabel = [0] * len(self.predict)
        for i in range(0, 4, 2):
            if self.rawlabel[i] != -1:
                framelabel[int(self.rawlabel[i]):int(self.rawlabel[i+1]+1)] \
                = [1] * (int(self.rawlabel[i+1])-int(self.rawlabel[i]) + 1)
        newlabel = []
        output_seg = []
        start = 0
        for l in self.length:
            if l == 0:
                break
            newlabel.append(framelabel[start])
            output_seg.append(self.predict[start])
            start += l.item()
        output_seg = np.array(output_seg)
        newlabel = np.array(newlabel)
        x = downsample.T[0]
        y = downsample.T[1]
        normal_index = np.nonzero(newlabel == 0)[0]
        anomaly_index = np.nonzero(newlabel != 0)[0]
        plt.title(self.title)
        plt.scatter(x=x[anomaly_index], y=y[anomaly_index], c=output_seg[anomaly_index], marker='x', label='Anomaly')
        plt.scatter(x=x[normal_index], y=y[normal_index], c=output_seg[normal_index], marker='o', label='Normal')
        cbar = plt.colorbar()
        plt.clim(0, 1)
        cbar.set_label("Predict Score")
        plt.legend()
        return figure
    def predictplot(self):
        figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios':[2, 1]})
        ax1.set_title(self.title)
        ax1.axis('off')
        ax2.plot(self.predict, label='Predict', lw=2)
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Frame number')
        ax2.set_ylabel('Anomaly score')
        ax2.set_xticks([0, len(self.predict)-1])
        ticks = [0]
        for i in range(0, len(self.rawlabel), 2):
            if self.rawlabel[i] != -1:
                ax2.add_patch(Rectangle((self.rawlabel[i], 0)
                    , self.rawlabel[i+1]-self.rawlabel[i], 1, color='red', alpha=0.2))
                #ticks.append(self.rawlabel[i])
                #ticks.append(self.rawlabel[i+1])
        for i in range(0, len(self.predict), len(self.predict)//10):
            ticks.append(i)
        #ticks.append(len(self.predict)-1)
        ax2.set_xticks(ticks)
        #try:
        #    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        #    text = "AUC: " + str(self.auc())[:4]
        #    plt.text(0, 1.05, text, bbox=props)
        #except:
        #    pass
        return figure

class AnomalyResult():
    def __init__(self):
        self.videos = {}
        self.types = {}
        self.scorer = Scorer()
        self.bagscorer = Scorer()
    def add(self, title, feature, predict, rawlabel, length):
        #compute label
        label = [0] * len(predict)
        for i in range(0, 4, 2):
            if rawlabel[i].item() != -1:
                start = max(0, rawlabel[i].item()-1)
                end = min(len(predict), rawlabel[i+1].item())
                label[start:end] = [1] * (end - start)
                if len(label) != len(predict):
                    ipdb.set_trace()
        #detect category
        category = title[:-3]
        if category.find("Normal") >= 0:
            category = "Normal"
        if category not in self.types:
            self.types[category] = AnomalyType(category)
        #add to predictions
        self.videos[title] = AnomalyVideo(title, feature, predict, label, rawlabel, length)
        self.types[category].add(predict, label)
        self.scorer.add(predict, label)
    def addbag(self, predict, label):
        self.bagscorer.add(predict, label)
    def auc(self):
        return self.scorer.auc()
    def aucbag(self):
        return self.bagscorer.auc()
    def roccurve(self):
        def draw(scorer, title='ROC curve'):
            fpr, tpr, t = roc_curve(scorer.label, scorer.predict)
            figure, ax = plt.subplots()
            plt.plot(fpr, tpr, color='red',
                             lw=2, label='Ours')
            for other in glob.glob('data/ROC/*'):
                mat = np.load(other, allow_pickle=True)
                name = os.path.basename(other).split(".")[0]
                if name.find('et al') > 0:
                    name += '.'
                #fpr, tpr, t = roc_curve(mat.T[1], mat.T[0])
                plt.plot(mat.T[0], mat.T[1], lw=1, ls='--', label=name)

            
            #plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xticks(np.arange(0., 1.01, step=0.1))
            plt.yticks(np.arange(0., 1.01, step=0.1))
            plt.grid(alpha=0.2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            return figure
        #return draw(self.scorer), draw(self.bagscorer, 'Bag ROC curve')
        return draw(self.scorer)
    def clusterplot(self, title):
        #figure = self.videos[title].clusterplot()
        #figure.savefig(title + "_features.png")
        return self.videos[title].clusterplot()
    def predictplot(self, title):
        return self.videos[title].predictplot()
