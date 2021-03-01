import torch
import cv2
import csv
import numpy as np
from datetime import datetime
import logging
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import src.config as config 
import glob
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


class logger():
    def __init__(self, args):
        self.args = args
        self.name, self.log = logfile(args)
        if args.savelog:
            if args.dataset == 'UCF':
                self.log.info("Saving log: {}".format(os.path.join(config.root_UCFCrime, "log", self.name + ".log")))
                self.root = os.path.join(config.root_UCFCrime, 'log', self.name)
            else:
                self.log.info("Saving log: {}".format(os.path.join(config.root_SH, "log", self.name + ".log")))
                self.root = os.path.join(config.root_SH, 'log', self.name)
            try:
                os.mkdir(self.root)
                self.log.info("create folder: {}".format(self.root))
            except:
                self.log.warning("folder already exists: {}".format(self.root))
    def recordparameter(self):
        self.log.info("loss = {0[0]}*bag+{0[1]}*innerbag+{0[2]}*maxmin+{0[3]}*smooth+{0[4]}*small".format(config.loss_parameter))
    def recordloss(self, losses, epoch):
        self.log.info(
                "bag: {:.4f}, smooth: {:.4f}, small: {:.4f}".format(losses[0], losses[1], losses[4])
        )
        self.log.info(
                "innerbag: {:.4f}, maxmin: {:.4f}".format(losses[3], losses[2])
        )

    def recordauc(self, result, epoch):
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
        if self.args.dataset == 'UCF':
            path = os.path.join(config.root_UCFCrime, 'image', self.name, path)
        else:
            path = os.path.join(config.root_SH, 'image', self.name, path)
        if not os.path.exists(os.path.dirname(path)):
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
        if args.dataset == 'UCF':
            configroot = config.root_UCFCrime
        else:
            configroot = config.root_SH
        logging.basicConfig(
            format=FORMAT,
            datefmt=DATE_FORMAT,
            level=logging.INFO, 
            handlers = [
                logging.FileHandler(os.path.join(configroot, "log", root + ".log"), 'a', 'utf-8'),
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
