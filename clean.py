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
    parser.add_argument('--list', action='store_true')
    return parser.parse_args()

def deletetarget(t):
    logfolder = os.path.join(root, 'log', t)
    logfile = os.path.join(root, 'log', t + '.log')
    writer = os.path.join('runs', t)
    if os.path.exists(logfolder):
        shutil.rmtree(logfolder)
        print("delete folder: {}".format(logfolder))
    if os.path.exists(logfile):
        os.unlink(logfile)
        print("delete logfile: {}".format(logfile))
    if os.path.exists(writer):
        shutil.rmtree(writer)
        print("delete writer: {}".format(writer))

if __name__ == '__main__':
    args = parse_args()
    root = config.root
    if len(args.target) > 0:
        deletetarget(args.target)
    else:
        folderlist = glob.glob(os.path.join(root, 'log/*.log'))
        namelist = [os.path.basename(x).split('.')[0] for x in folderlist]
        target = namelist[-1 * args.count:]
        if args.list:
            for n in folderlist:
                print(n)
            exit(0)
        for t in target:
            deletetarget(t)
