import cv2
import ipdb
import numpy as np
import imutils
import os
import argparse
import sys
from src import config

class converter():
    def __init__(self, path):
        self.videopath = path
        self.dir = os.path.join(os.path.dirname(
            path), os.path.basename(path).split(".")[0])
        isdir = self.mkdir(self.dir)

    def mkdir(self, path):
        if os.path.isdir(path):
            return 0
        else:
            os.system("mkdir " + path)
            return 1

    def reset(self, args):
        target = self.dir
        if args.frame:
            os.system("rm -rf " + os.path.join(target, 'frame'))


    def toframe(self):
        path = os.path.join(self.dir, 'frame')
        proceed = self.mkdir(path)
        if proceed==0:
            return
        vidcap = cv2.VideoCapture(self.videopath)
        success, image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite(self.dir + "/frame/%d.jpg" %
                        count, image)     # save frame as JPEG file
            success, image = vidcap.read()
            count += 1
        vidcap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="reset folders")
    parser.add_argument("--frame", action="store_true")
    args = parser.parse_args()
    root = config.root_UCFCrime
    data_list = np.genfromtxt(os.path.join(root, "Anomaly_Train.txt"), dtype=str)
    for video in data_list:
        if video.find("Normal") >= 0:
            convert = converter(os.path.join(root, video))
        else:
            convert = converter(os.path.join(root, "Anomaly-Videos", video))

        if args.reset:
            convert.reset(args)
        else:
            print(video)
            convert.toframe()
    data_list_test = np.genfromtxt(
            os.path.join(root, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"), dtype=str
    )
    for data in data_list_test:
        video, category, _, _, _, _ = data
        if category == 'Normal':
            convert = converter(os.path.join(root, 'Testing_Normal_Videos_Anomaly', video))
        else:
            convert = converter(os.path.join(root, 'Anomaly-Videos', category, video))
        if args.reset:
            convert.reset(args)
        else:
            print(video)
            convert.toframe()

        
