import glob
import src.config as config
import os
import datetime
import ipdb
import argparse
import shutil
 
def parse_args():
    parser = argparse.ArgumentParser(description="clean")
    parser.add_argument('--count', type=int, default=1)
    parser.add_argument('--target', type=str, default='')
    parser.add_argument('--predicttarget', type=str, default='')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--rank', action='store_true')
    parser.add_argument('--deletelast', action='store_true')
    parser.add_argument('--find', type=str, default='')
    parser.add_argument('--killmode', action='store_true')
    parser.add_argument('--predictnewest', action='store_true')
    return parser.parse_args()

def getpath(t):
    logfolder = os.path.join(root, 'log', t)
    logfile = os.path.join(root, 'log', t + '.log')
    writer = os.path.join('runs', t)
    return logfolder, logfile, writer

def getscore(lines):
    aucs = []
    for l in lines:
        if l.find("AUC") > 0:
            aucs.append(float(l.split("AUC: ")[1]))
    return aucs

def getmodel(lines):
    models = []
    for i, l in enumerate(lines):
        if l.find("Save model:") > 0 and l.find("C3D") < 0 and l.find("attn") < 0:
            path = l.split("Save model: ")[1][:-1]
            models.append(path)
    return models
        

def getnote(lines):
    note = ''
    if lines[3].find("Note") > 0:
        note = lines[4].split("INFO: ")[1].replace("\n", "")
    return note

def getloss(lines):
    loss = ''
    for l in lines:
        if l.find('loss = ') > 0:
            loss = l.split("= ")[1][:-1]
            break
    return loss
def predict(modelpath):
    command = "python predict.py --gpus 0 --load_backbone --model_path " + modelpath
    print(command)
    os.system(command)

class result():
    def __init__(self, name):
        self.name = name
        self.logfolder, self.logfile, self.writer = getpath(name)
        self.lines = open(self.logfile).readlines()
        self.scores = getscore(self.lines)
        self.model_path = getmodel(self.lines)
        self.epoch = len(self.scores)
        self.note = getnote(self.lines)
        self.loss = getloss(self.lines)
    def __gt__(self, other):
        assert self.epoch > 0 and other.epoch > 0, 'experiment ungoing'
        return max(self.scores) > max(other.scores)
    def __str__(self):
        if self.epoch > 0:
            return "{}\
                    \nEpoch: {}\
                    \nBest: {}\
                    \nNote: {}\
                    \nLoss: {}\
                    \n".format(self.logfolder, len(self.scores), max(self.scores), self.note, self.loss)
        else:
            return "{}\
                    \nEpoch: 0\
                    \nBest: -\
                    \nNote: {}\
                    \nLoss: {}\
                    \n".format(self.logfolder, self.note, self.loss)
    def clean(self):
        print("Target:")
        print(self)
        confirm = input("Sure to delete?\n")
        if confirm in ['y', 'Y', 'yes', 'YES', 'Yes', "sure"]:
            if os.path.exists(self.logfolder):
                shutil.rmtree(self.logfolder)
                print("delete folder: {}".format(self.logfolder))
            if os.path.exists(self.logfile):
                os.unlink(self.logfile)
                print("delete logfile: {}".format(self.logfile))
            if os.path.exists(self.writer):
                shutil.rmtree(self.writer)
                print("delete writer: {}".format(self.writer))
        else:
            print("Target unchanged")

if __name__ == '__main__':
    args = parse_args()
    root = config.root
    folderlist = glob.glob(os.path.join(root, 'log/*.log'))
    namelist = [os.path.basename(x).split('.')[0] for x in folderlist]
    namelist = sorted(namelist)
    results = [result(n) for n in namelist]
    if args.list:
        for result in results:
            print(result)
    if args.rank:
        for result in sorted(results):
            print(result)
    if len(args.target) > 0:
        for result in results:
            if args.target == result.logfolder or args.target == result.name:
                result.clean()
    if args.deletelast:
        results[-1].clean()
    if args.find:
        for result in results:
            if result.note.find(args.find) >= 0:
                print(result)
    if args.killmode:
        minepoch = input("\nKill results under _ epoch?\n")
        pathresult = {}
        for result in results:
            pathresult[result.logfolder] = result
            if result.epoch <= int(minepoch):
                result.clean()
        while True:
            path = input("\nInput the path to kill\n")
            if path in pathresult:
                pathresult[path].clean()
            else:
                break
    if len(args.predicttarget) > 0:
        target = None
        for result in results:
            if args.predicttarget == result.logfolder or args.predicttarget == result.name:
                target = result
        assert target != None, 'Target Not Found'
        #index = target.scores.index(max(target.scores))
        #modelpath = target.model_path[index]
        #predict(modelpath)
        for i, path in enumerate(target.model_path):
            predict(path)
            print("Formal Score: " + str(target.scores[i]))
            print("")

    if args.predictnewest:
        target = results[-1]
        assert target.epoch > 0, "Training incomplete"
        index = target.scores.index(max(target.scores))
        modelpath = target.model_path[index]
        modelpath = os.path.join(target.logfolder, str(target.epoch - 1) + '.pth')
        predict(modelpath)

            
