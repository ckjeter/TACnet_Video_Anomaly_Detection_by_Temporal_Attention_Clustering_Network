import torch
import csv
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter 
import logging
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, auc
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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
    def __init__(self, title, feature, predict, label, rawlabel):
        super().__init__()
        self.title = title
        self.feature = feature
        self.predict = predict
        self.label = label
        self.rawlabel = rawlabel
    def clusterplot(self):
        figure, ax = plt.subplots()
        downsample = TSNE(n_components=2).fit_transform(self.feature.squeeze(0).cpu().detach())
        newlabel = [0] * self.feature.shape[1]
        for i in range(0, 4, 2):
            if self.rawlabel[i] != -1:
                newlabel[int(self.rawlabel[i]/16):int(self.rawlabel[i+1]/16)] \
                = [1] * (int(self.rawlabel[i+1]/16)-int(self.rawlabel[i]/16))
        output_seg = []
        for i in range(0, len(self.predict), 16):
            output_seg.append(self.predict[i])
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
        figure, ax = plt.subplots()
        plt.plot(self.predict)
        plt.title(self.title)
        plt.ylim([0, 1])
        plt.xlabel('Frame number')
        plt.ylabel('Anomaly score')
        for i in range(0, len(self.rawlabel), 2):
            if self.rawlabel[i] != -1:
                ax.add_patch(Rectangle((self.rawlabel[i], 0)
                    , self.rawlabel[i+1]-self.rawlabel[i], 1, color='red', alpha=0.5))
        try:
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            text = "AUC: " + str(self.auc())[:4]
            plt.text(0, 1.1, text, bbox=props)
        except:
            pass
        return figure

class AnomalyResult():
    def __init__(self):
        self.videos = {}
        self.types = {}
        self.scorer = Scorer()
        self.bagscorer = Scorer()
    def add(self, title, feature, predict, rawlabel):
        #compute label
        label = [0] * len(predict)
        for i in range(0, 4, 2):
            if label[i] != -1:
                label[rawlabel[i]:rawlabel[i+1]] = [1] * (rawlabel[i+1]-rawlabel[i])
        #detect category
        category = title[:-3]
        if category.find("Normal") >= 0:
            category = "Normal"
        if category not in self.types:
            self.types[category] = AnomalyType(category)
        #add to predictions
        self.videos[title] = AnomalyVideo(title, feature, predict, label, rawlabel)
        self.types[category].add(predict, label)
        self.scorer.add(predict, label)
    def addbag(self, predict, label):
        self.bagscorer.add(predict, label)
    def auc(self):
        return self.scorer.auc()
    def aucbag(self):
        return self.bagscorer.auc()
    def roccurve(self):
        fpr, tpr, t = roc_curve(self.scorer.label, self.scorer.predict)
        figure, ax = plt.subplots()
        plt.plot(fpr, tpr, color='darkorange',
                         lw=2, label='ROC curve (area = %0.2f)' % self.scorer.auc())
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        return figure
    def clusterplot(self, title):
        #figure = self.videos[title].clusterplot()
        #figure.savefig(title + "_features.png")
        return self.videos[title].clusterplot()
    def predictplot(self, title):
        return self.videos[title].predictplot()



class logger():
    def __init__(self, args):
        self.args = args
        self.root, self.log = logfile(args)
        if args.savelog:
            self.log.info("Saving log: {}".format(os.path.join("log", self.root + ".log")))
            self.root = os.path.join('log', self.root)
            try:
                os.mkdir(self.root)
            except:
                self.log.warning("create folder failed")
            if len(args.root) > 0:
                self.writer = SummaryWriter(os.path.join('runs', args.root))
            else:
                self.writer = SummaryWriter(os.path.join('runs', self.root.split("/")[1]))

    def recordloss(self, losses, epoch):
        if self.args.savelog: 
            self.writer.add_scalar('bag_loss', losses[0], epoch)
            self.writer.add_scalar('cluster_loss_far', losses[1], epoch)
            self.writer.add_scalar('cluster_loss_close', losses[2], epoch)
            self.writer.add_scalar('smooth_loss', losses[3], epoch)
            self.writer.add_scalar('inner_anomaly_loss', losses[4], epoch)
            self.writer.add_scalar('inner_normal_loss', losses[5], epoch)
        self.log.info(
                "bag_loss: {:.4f}, smooth_loss: {:.4f}".format(losses[0], losses[3])
        )
        self.log.info(
                "cluster_far_loss: {:.4f}, cluster_close_loss: {:.4f}".format(losses[1], losses[2])
        )
        self.log.info(
                "innerbag_anomaly_loss: {:.4f}, innerbag_normal_loss: {:.4f}".format(losses[4], losses[5])
        )

    def recordauc(self, result, epoch):
        if self.args.savelog:
            self.writer.add_scalar('bag_accuracy', result.aucbag(), epoch)
            self.writer.add_scalar('AUC', result.auc(), epoch)
        self.log.info("Bag Acc: {:.4f}".format(result.aucbag()))
        self.log.info("AUC: {:.4f}".format(result.auc()))
    def auc_types(self, result):
        for k, v in result.types.items():
            if k != "Normal":
                print(k, v.auc(), v.count)
            else:
                print(k, v.count)
    def savemodel(self, net, epoch):
        savepath = os.path.join(self.root, str(epoch) + '.pth')
        self.log.info("    Save model: {}".format(savepath))
        torch.save(net.state_dict(), savepath)

    def info(self, message):
        self.log.info(message)

def logfile(args):
    if len(args.root) > 0:
        root = args.root
    else:
        root = str(datetime.now().strftime('%m%d_%H:%M'))
    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%m/%d_%H:%M:%S'
    if args.savelog:
        logging.basicConfig(
            format=FORMAT,
            datefmt=DATE_FORMAT,
            level=logging.INFO, 
            handlers = [
                logging.FileHandler(os.path.join("log", root + ".log"), 'w', 'utf-8'),
                logging.StreamHandler()
            ],
        )
    else:
        logging.basicConfig(
            format=FORMAT,
            datefmt=DATE_FORMAT,
            level=logging.INFO, 
            handlers = [
                logging.StreamHandler()
            ],
        )
    logger = logging.getLogger()
    return root, logger

        
