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
import os
import ipdb

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v

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
    def __init__(self, title, feature, predict, label, length):
        super().__init__()
        self.title = title
        self.feature = feature
        self.predict = predict
        self.label = label
        self.length = length
    def clusterplot(self):
        figure, ax = plt.subplots()
        downsample = TSNE(n_components=2).fit_transform(self.feature.squeeze(0).cpu().detach())
        newlabel = []
        output_seg = []
        start = 0
        for l in self.length:
            if l == 0:
                break
            newlabel.append(self.label[start])
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
        ax1.set_title(self.title.split(".")[0])
        ax1.axis('off')
        ax2.plot(self.predict, label='Predict', lw=2)
        #ax2.title(self.title)
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Frame number')
        ax2.set_ylabel('Anomaly score')
        ax2.set_xticks([0, len(self.predict)-1])
        ticks = []

        flag = 0
        start = -1
        for i in range(len(self.label)):
            if self.label[i] > flag: #0 -> 1
                start = i
                flag = self.label[i]
            if self.label[i] < flag: #1 -> 0
                ax2.add_patch(Rectangle((start, 0)
                    , i - start, 1, color='red', alpha=0.2))
                #ticks.append(self.rawlabel[i])
                #ticks.append(self.rawlabel[i+1])
                flag = 0
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
    def add(self, title, feature, predict, label, length):
        #compute label
        #detect category
        category = title.split("_")[0]
        self.types[category] = AnomalyType(category)
        #add to predictions
        self.videos[title] = AnomalyVideo(title, feature, predict, label, length)
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
            plt.plot(fpr, tpr, color='darkorange',
                             lw=2, label='ROC curve (area = %0.2f)' % scorer.auc())
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
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
