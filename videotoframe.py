import cv2
import ipdb
import numpy as np
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

    def reset(self):
        os.system("rm -rf " + self.dir)

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

    def BG_sub(self, mode='MOG2'):
        path = os.path.join(self.dir, mode)
        self.mkdir(path)
        vidcap = cv2.VideoCapture(self.videopath)
        if mode == 'MOG2':
            backSub = cv2.createBackgroundSubtractorMOG2()
        else:
            backSub = cv2.createBackgroundSubtractorKNN()
        count = 0
        while True:
            ret, frame = vidcap.read()
            if frame is None:
                break
            fgMask = backSub.apply(frame)
            #cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            #cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            frame = cv2.bitwise_and(frame, frame, mask=fgMask)
            cv2.imwrite(os.path.join(self.dir, mode, "%d.jpg" % count), frame)
            count += 1
        vidcap.release()        
    def moving_average(self):
        self.mkdir('moving')
        vidcap = cv2.VideoCapture(self.videopath)
        ret, frame = vidcap.read()
        avg = cv2.blur(frame, (4, 4))
        avg_float = np.float32(avg)

        count = 0
        while(1):
            ret, frame = vidcap.read()
            if ret == False:
                break
            blur = cv2.blur(frame, (4, 4))
            diff = cv2.absdiff(avg, blur)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            thresh = cv2.morphologyEx(
                thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            cntImg, cnts, _ = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in cnts:
                if cv2.contourArea(c) < 500:
                    continue
                # if detect anything...
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
            cv2.imwrite(self.dir + "/moving/%d.png" %
                        count, frame)     # save frame as JPEG file
            cv2.accumulateWeighted(blur, avg_float, 0.01)
            avg = cv2.convertScaleAbs(avg_float)
            count += 1
        vidcap.release()

    def optical(self):
        self.mkdir('opt')
        vidcap = cv2.VideoCapture(self.videopath)
        ret, frame1 = vidcap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        count = 0
        while(1):
            ret, frame2 = vidcap.read()
            if not ret:
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(
                prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(self.dir + "/opt/%d.png" %
                        count, rgb)     # save frame as JPEG file
            prvs = next
            count += 1
        vidcap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="reset folders")
    args = parser.parse_args()
    root = config.root
    data_list = np.genfromtxt(os.path.join(root, "Anomaly_Train.txt"), dtype=str)
    for video in data_list:
        convert = converter(os.path.join(root, video))
        if args.reset:
            convert.reset()
        else:
            print(video)
            convert.BG_sub("KNN")
            #convert.toframe()
        break
