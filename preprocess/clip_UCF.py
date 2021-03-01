from src.dataset import *
import glob
import cv2
import os
import ipdb
from tqdm import tqdm

def gettitle(video):
    video = os.path.basename(video)
    if video.find("Normal") >= 0:
        title = video.split("_x")[0]
    else:
        title = video.split("_")[0]
    return title

def path_generate(root, video):
    if video.find("Normal") >= 0:
        input_path = os.path.join(root, video.split(".")[0], 'frame')
    else:
        input_path = os.path.join(root, "Anomaly-Videos", video.split(".")[0], 'frame')
    return input_path

def clip(folder_dir):
    framelist = sorted(glob.glob(folder_dir + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
    if len(framelist) < 512:
        cv2.imwrite('.black.jpg', np.zeros((112, 112, 3), dtype=np.float32))
        black = '.black.jpg'
        gap = 512 - len(framelist)
        framelist += [black] * (512 - len(framelist))
        framelist = np.array(framelist)
        clips = np.array_split(framelist, 32)
        length = [len(c) for c in clips]
        for i in range(31, 0, -1):
            if gap < 16:
                length[i] -= gap
                break
            else:
                gap -= length[i]
                length[i] = 0
    else:
        framelist = np.array(framelist)
        clips = np.array_split(framelist, 32)
        length = [len(c) for c in clips]
    sliceclips = []
    for c in clips:
        try:
            clip = [cc[0] for cc in np.array_split(c, 16)]
        except:
            clip = c
        imgs = np.array([np.array(Image.open(path).resize((112, 112), Image.BICUBIC), dtype=np.float32) for path in clip])
        sliceclips.append(imgs)
    sliceclips = np.stack(sliceclips, axis=0)
    return sliceclips, np.array(length)

def generate_clip(mode):
    video_paths = []
    label = []
    title = []
    if mode.find('train')>=0:
        datalist = np.genfromtxt(
                os.path.join(root, "Anomaly_Train.txt"), dtype=str)
        for video in datalist:
            input_path = path_generate(root, video)
            title.append(gettitle(video))
            video_paths.append(input_path)
    else:
        datalist = np.genfromtxt(
                os.path.join(
                    config.root, "Temporal_Anomaly_Annotation_for_Testing_Videos.txt"), dtype=str)
        for data in datalist:
            video, category, f1, f2, f3, f4 = data
            title.append(gettitle(video))
            if category == 'Normal':
                video = os.path.join("Testing_Normal_Videos_Anomaly", video)
            else:
                video = os.path.join(category, video)
            input_path = path_generate(root, video)
            video_paths.append(input_path)
            label.append(np.array([int(f1), int(f2), int(f3), int(f4)]))
    for i, path in enumerate(tqdm(video_paths)):
        imgs, length = clip(path)
        targetfolder = os.path.join(root, mode, title[i])
        framepath = os.path.join(targetfolder, 'frame')
        os.makedirs(framepath, exist_ok=True)

        seq_length, clip_length, h, w, channel = imgs.shape
        realcount = 0
        for seq_count in range(imgs.shape[0]):
            for order in range(imgs.shape[1]):
                frame = imgs[seq_count][order]
                step = max(1, length[seq_count] // 16)
                count = sum(length[:seq_count]) + (step * order)

                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                cv2.imwrite(os.path.join(framepath, '%d.jpg' % realcount), frame)
                realcount += 1
        if mode.find('train') >= 0:
            if title[i].find("Normal") >= 0:
                label = 0
            else:
                label = 1
            output_file = np.array((imgs, label, length), dtype=object)
        else:
            output_file = np.array((imgs, label[i], length), dtype=object)
        output_path = os.path.join(targetfolder, title[i] + '.npy')
        #np.save(output_path, output_file, allow_pickle=True)

root = config.root_UCFCrime 
generate_clip('train')
generate_clip('test')
