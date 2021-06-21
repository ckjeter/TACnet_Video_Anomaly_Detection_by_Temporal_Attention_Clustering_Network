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
        if args.BG_sub:
            os.system("rm -rf " + os.path.join(target, 'BG_sub'))
        if args.optical:
            os.system("rm -rf " + os.path.join(target, "optical"))
        if args.moving:
            os.system("rm -rf " + os.path.join(target, "moving"))


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

    def BG_sub(self, mode='KNN'):
        path = os.path.join(self.dir, "BG_sub")
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
            #cv2.rectangle(*frame, (10, 2), (100,20), (255,255,255), -1)
            #cv2.putText(frame, str(vidcap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
            #           cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            mask = cv2.bitwise_and(frame, frame, mask=fgMask)
            cnts = cv2.findContours(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            mask_all = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < 500:
                    continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                mask_all[y:y+h, x:x+w, :] = 255
            mask = cv2.bitwise_and(frame, frame, mask=mask_all)
            frame = cv2.GaussianBlur(frame, (21, 21), 0)
            BG = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask_all, mask_all))
            frame = cv2.add(mask, BG)
            cv2.imwrite(os.path.join(self.dir, 'BG_sub', "%d.jpg" % count), frame)
            count += 1
        vidcap.release()        
    def moving_average(self):
        path = os.path.join(self.dir, "moving")
        self.mkdir(path)
        vidcap = cv2.VideoCapture(self.videopath)
        firstFrame = None
        count = 0
        while(1):
            ret, frame = vidcap.read()
            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if frame is None:
                break
            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            # if the first frame is None, initialize it
            if firstFrame is None:
                firstFrame = gray
                firstFrame = cv2.blur(firstFrame, (4, 4))
                avg_float = np.float32(firstFrame)
                continue
            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # loop over the contours
            for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < 500:
                    continue
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(self.dir, 'moving', "%d.jpg" % count), frame)
            cv2.accumulateWeighted(gray, avg_float, 0.01)
            firstFrame = cv2.convertScaleAbs(avg_float)
            count += 1
        vidcap.release()
    def saliency(self):
        path = os.path.join(self.dir, 'saliency')
        self.mkdir(path)
        vidcap = cv2.VideoCapture(self.videopath)
        saliency = None
        count = 0
        while True:
            ret, frame = vidcap.read()
            if frame is None:
                break
            if saliency is None:
                saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
                saliency.setImagesize(frame.shape[1], frame.shape[0])
                saliency.init()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            (success, saliencyMap) = saliency.computeSaliency(gray)
            saliencyMap = (saliencyMap * 255).astype("uint8")
            
            cv2.imwrite(os.path.join(path, '%d.jpg' % count), saliencyMap)
            count += 1
        vidcap.release()


    def optical(self):
        path = os.path.join(self.dir, 'optical')
        self.mkdir(path)
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
    parser.add_argument("--frame", action="store_true")
    parser.add_argument("--BG_sub", action="store_true")
    parser.add_argument("--optical", action="store_true")
    parser.add_argument("--moving", action="store_true")
    parser.add_argument("--saliency", action="store_true")
    args = parser.parse_args()
    root = config.root
    data_list = np.genfromtxt(os.path.join(root, "Anomaly_Train.txt"), dtype=str)
    '''
    for video in data_list:
        if video.find("Normal") >= 0:
            convert = converter(os.path.join(root, video))
        else:
            convert = converter(os.path.join(root, "Anomaly-Videos", video))

        if args.reset:
            convert.reset(args)
        else:
            print(video)
            if args.frame:
                convert.toframe()
            if args.BG_sub:
                convert.BG_sub()
            if args.optical:
                convert.optical()
            if args.moving:
                convert.moving_average()
            if args.saliency:
                convert.saliency()
    '''
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
            if args.frame:
                convert.toframe()
            if args.BG_sub:
                convert.BG_sub()
            if args.optical:
                convert.optical()
            if args.moving:
                convert.moving_average()
            if args.saliency:
                convert.saliency()
        
