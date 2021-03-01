import glob
import numpy as np
import cv2
import os
import ipdb
from tqdm import tqdm
from PIL import Image
import src.config as config

root = '/data/dataset/ShanghaiTech'
train_list = set(np.genfromtxt("SH_Train.txt", dtype=str))
test_list = set(np.genfromtxt("SH_Test.txt", dtype=str))

def clip(folder_dir):
    framelist = sorted(glob.glob(folder_dir + "/*"), key=lambda x : int(os.path.basename(x).split(".")[0]))
    cv2.imwrite('.black.jpg', np.zeros((112, 112, 3), dtype=np.float32))
    black = '.black.jpg'
    #if len(framelist) < 512:
    #    gap = 512 - len(framelist)
    #    framelist += [black] * (512 - len(framelist))
    #    framelist = np.array(framelist)
    #    clips = np.array_split(framelist, 32)
    #    length = [len(c) for c in clips]
    #    for i in range(31, 0, -1):
    #        if gap < 16:
    #            length[i] -= gap
    #            break
    #        else:
    #            gap -= length[i]
    #            length[i] = 0
    #else:
    #    framelist = np.array(framelist)
    #    clips = np.array_split(framelist, 32)
    #    length = [len(c) for c in clips]
    #    ipdb.set_trace()

    framelist = np.array(framelist)
    clips = np.array_split(framelist, 32)
    length = [len(c) for c in clips]
    sliceclips = []
    for c in clips:
        if len(c) >= 16:
            clip = [cc[0] for cc in np.array_split(c, 16)]
        else:
            clip = []
            repeat = int(16/len(c))
            for i in range(len(c)):
                if len(c) - i - 1< 16 % len(c):
                    clip += [c[i]] * (repeat + 1) 
                else:
                    clip += [c[i]] * repeat

        imgs = np.array([np.array(Image.open(path).resize((112, 112), Image.BICUBIC), dtype=np.float32) for path in clip])
        sliceclips.append(imgs)
    sliceclips = np.stack(sliceclips, axis=0)
    return sliceclips, np.array(length)

def generate_saliency(source):
    root = config.root_SH
    framecount = 0
    video_paths = []
    labels = []
    title = []
    if source.find('train')>=0:
        datalist = glob.glob(os.path.join(root, 'training/frames/*'))
        for path in datalist:
            video_paths.append(path)
            title.append(os.path.basename(path))
    else:
        datalist = glob.glob(os.path.join(root, 'testing/frames/*'))
        for path in datalist:
            video_paths.append(path)
            title.append(os.path.basename(path))
            label_path = path.replace("frames", "test_frame_mask")+'.npy'
            labels.append(np.load(label_path))
    for i, path in enumerate(tqdm(video_paths)):
        assert title[i] in train_list or title[i] in test_list, "File not found"
        imgs, length = clip(path)
        if title[i] in train_list:
            output_path = os.path.join(root, 'train', title[i]+'.npy')
            if source.find('train') >= 0:
                label = 0
            else:
                label = 1
        elif title[i] in test_list:
            output_path = os.path.join(root, 'test', title[i]+".npy")
            if source.find('train') >= 0:
                label = np.zeros(length.sum())
            else:
                label = labels[i]
        else:
            ipdb.set_trace()
        output_file = np.array((imgs, label, length), dtype=object)
        np.save(output_path, output_file, allow_pickle=True)
    print(framecount / len(title))

generate_saliency('test')
generate_saliency('train')
