import cv2
import ipdb
import numpy as np
import glob
import imutils
import os
import argparse
import sys

class converter():
    def __init__(self, path):
        self.videopath = path
        self.dir = os.path.join(os.path.dirname(path), os.path.basename(path).split(".")[0])
        self.dir = self.dir.replace("videos", "frames")
        isdir = self.mkdir(self.dir)
        self.count = None

    def mkdir(self, path):
        if os.path.isdir(path):
            return 0
        else:
            os.system("mkdir " + path)
            return 1

    def reset(self, args):
        target = self.dir
        if args.frame:
            os.system("rm -rf " + os.path.join(self.dir))

    def toframe(self):
        vidcap = cv2.VideoCapture(self.videopath)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(os.path.join(self.dir, "%03d.jpg" % count), image)     # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        vidcap.release()
        self.count = count
        print(self.dir, count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="reset folders")
    parser.add_argument("--frame", action="store_true")
    args = parser.parse_args()
    root = config.root_SH
    datalist = glob.glob(os.path.join(root, 'training/videos/*')) 
    framecount = {}
    for path in datalist:
        convert = converter(path)
        convert.toframe()
        framecount[convert.videopath] = convert.count
    counts = framecount.values()
    video = framecount.keys()
    average = (float)(sum(counts) / len(counts))
    maxcount = max(counts)
    maxvideos = video[counts.index(maxcount)]
    mincount = min(counts)
    minvideos = video[counts.index(mincount)]
    print("Average length: {}".format(average))
    print("Max length : {}".format(maxcount))
    print(maxvideos)
    print("Min length : {}".format(mincount))
    print(minvideos)
        
