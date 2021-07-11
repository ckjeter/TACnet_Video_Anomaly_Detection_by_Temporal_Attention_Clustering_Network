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
        return draw(self.scorer), draw(self.bagscorer, 'Bag ROC curve')
    def clusterplot(self, title):
        #figure = self.videos[title].clusterplot()
        #figure.savefig(title + "_features.png")
        return self.videos[title].clusterplot()
    def predictplot(self, title):
        return self.videos[title].predictplot()



class logger():
    def __init__(self, args):
        self.args = args
        self.name, self.log = logfile(args)
        if args.savelog:
            self.log.info("Saving log: {}".format(os.path.join(config.root, "log", self.name + ".log")))
            self.root = os.path.join(config.root, 'log', self.name)
            try:
                os.mkdir(self.root)
                self.log.info("create folder: {}".format(self.root))
            except:
                self.log.warning("folder already exists: {}".format(self.root))
            writerpath = os.path.join('runs', self.name)
            self.writer = SummaryWriter(writerpath)
            self.log.info("create writer: {}".format(writerpath))
    def recordparameter(self):
        self.log.info("loss = {0[0]}*bag+{0[1]}*cluster+{0[2]}*innerbag+{0[3]}*maxmin+{0[4]}*smooth+{0[5]}*small".format(config.loss_parameter))
    def recordloss(self, losses, epoch):
        if self.args.savelog: 
            self.writer.add_scalar('bag_loss', losses[0], epoch)
            self.writer.add_scalar('cluster_loss', losses[1], epoch)
            self.writer.add_scalar('smooth_loss', losses[2], epoch)
            self.writer.add_scalar('inner_anomaly_loss', losses[4], epoch)
        self.log.info(
                "bag: {:.4f}, smooth: {:.4f}, small: {:.4f}".format(losses[0], losses[2], losses[5])
        )
        self.log.info(
                "cluster: {:.4f}, innerbag: {:.4f}, maxmin: {:.4f}".format(losses[1], losses[4], losses[3])
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
    def savemodel(self, model, epoch):
        backbone, net = model
        savepath_C3D = os.path.join(self.root, str(epoch) + 'C3D.pth')
        savepath_net = os.path.join(self.root, str(epoch) + '.pth')
        self.log.info("    Save model: {}".format(savepath_C3D))
        self.log.info("    Save model: {}".format(savepath_net))
        torch.save(backbone.state_dict(), savepath_C3D)
        torch.save(net.state_dict(), savepath_net)
    def info(self, message):
        self.log.info(message)
    def savefig(self, figure, path):
        '''
        folder = os.path.join('image', self.name)
        if not os.path.exists(folder):
            os.mkdir(folder)
        '''
        path = os.path.join(config.root, 'image', self.name, path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            figure.savefig(path)
            plt.close(figure)
        except:
            cv2.imwrite(path, figure)

def logfile(args):
    if len(args.root) > 0:
        root = args.root
    elif len(args.model_path) > 0:
        root = os.path.basename(os.path.dirname(args.model_path))
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
                logging.FileHandler(os.path.join(config.root, "log", root + ".log"), 'a', 'utf-8'),
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
