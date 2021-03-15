import torch
import csv
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter 
import logging
from sklearn.metrics import fbeta_score 
from sklearn.metrics import roc_auc_score
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

class AnomalyType():
    def __init__(self, category):
        self.category = category
        self.scorer = Scorer()
    def add(self, predict, label):
        self.scorer.add(predict, label)
    def auc(self):
        return self.scorer.auc()

class AnomalyScorer():
    def __init__(self):
        self.types = {}
        self.scorer = Scorer()
        self.bagscorer = Scorer()
    def add(self, title, predict, label):
        category = title[:-3]
        if category.find("Normal") >= 0:
            category = "Normal"
        if category not in self.types:
            self.types[category] = AnomalyType(category)
        self.types[category].add(predict, label)
        self.scorer.add(predict, label)
    def addbag(self, predict, label):
        self.bagscorer.add(predict, label)
    def auc(self):
        return self.scorer.auc()
    def aucbag(self):
        return self.bagscorer.auc()

class logger():
    def __init__(self, args):
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
        self.writer.add_scalar('bag_loss', losses[0], epoch)
        self.writer.add_scalar('cluster_loss_far', losses[1], epoch)
        self.writer.add_scalar('cluster_loss_close', losses[2], epoch)
        self.writer.add_scalar('smooth_loss', losses[3], epoch)
        self.log.info(
                "bag_loss: {:.4f}, smooth_loss: {:.4f}".format(losses[0], losses[3])
        )
        self.log.info(
                "cluster_far_loss: {:.4f}, cluster_close_loss: {:.4f}".format(losses[1], losses[2])
        )

    def recordauc(self, result, epoch):
        self.writer.add_scalar('bag_accuracy', result.aucbag(), epoch)
        self.log.info("Bag Acc: {:.4f}".format(result.aucbag()))
        self.writer.add_scalar('AUC', result.auc(), epoch)
        self.log.info("AUC: {:.4f}".format(result.auc()))
    def auc_types(self, result):
        for k, v in result.types.items():
            if k != "Normal":
                print(k, v.auc())
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

        
