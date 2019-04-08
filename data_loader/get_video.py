import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.utils import Sequence
import random
import glob

def get_video_frames(src, fpv, frame_height, frame_width):
    # print('reading video from', src)
    cap = cv2.VideoCapture(src)

    frames = []
    if not cap.isOpened():
        cap.open(src)

    ret, frame = cap.read()

    while(ret):
        frames.append(frame)
        # Capture frame-by-frame
        ret, frame = cap.read()

    # When everything done, release the capture
    cap.release()

    #print("frame length: ", len(frames))
    #if the amount of frames is less than the frames per video, set step to one
    step = len(frames)//fpv
    #gets which section of frames to select
    if not step == 0:
        frame_select = np.random.choice(np.arange(step))
    else:
        frame_select = 0
    #while theres not enough frames to fill up to fpv, create duplicates of each frame and add them after their original frame.
    while len(frames) < fpv:
        for i in range(len(frames)):
            filler = frames[i]
            frames.insert(i+1, filler)

    #grabs section of frames specified by frame_select. if 0 its 0->fpv, 1 is fpv->fpv * 2 etc...
    avg_frames = frames[(frame_select*fpv):(frame_select*fpv + fpv)]
    #print("avg frame length: ", len(avg_frames))
    avg_resized_frames = []
    for af in avg_frames:
        rsz_f = cv2.resize(af, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        rsz_f = rsz_f / 255.0 #normalize
        avg_resized_frames.append(rsz_f)
    #print("shape: ", np.asarray(avg_resized_frames).shape)
    return np.asarray(avg_resized_frames)

def get_video(video_path, frames_per_video, frame_height, frame_width):
    # Read clip and appropiately send the sports' class
    frames = get_video_frames(video_path.strip(), frames_per_video, frame_height, frame_width)

    frames = np.expand_dims(frame, axis=0)

    return frames


class Video_Meta_Loader(Sequence):
    """
    Video loader for meta-learning model. img_dir should point to directory containing images of a single class
    """
    def __init__(self, dir, frames_per_video, img_dim, class, batch_size):
        self.img_dim = img_dim
        if img_dir[-1] == "/":
            self.paths = glob.glob(img_dir + "*")
        else:
            self.paths = glob.glob(img_dir + "/*")
        self.fpv = frames_per_video
        self.classes = num_classes
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = np.vstack([get_video(video, self.fpv, self.img_dim[0], self.img_dim[1]) for video in batch])

        return batch, class
