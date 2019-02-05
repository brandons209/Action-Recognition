from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch
import cv2
import numpy as np
import glob

def get_frames(video_path, num_frames):
    video = cv2.VideoCapture(video_path)
    frames = []
    while video.isOpened():
        _, frame = video.read()-
        frames.append(frame)
    video.release()

    #get all frames from video
    frames = np.array(frames)
    #split frames into batches of num_frames
    frames = np.split(frames, num_frames)
    #randomly select consecutive num_frames batch from frames
    return np.random.choice(frames)

"""
video class, assumes each video is in a folder with its class name in the directory <path>
each call will return number specified in frames_per_batch, **should be a multiple of the framerate**
scales frames from images to vid_dim
returns consecutive frames of a video with its respective one-hot encoded label.
"""
class Videos(Dataset):

    def __init__(self, path, vid_dim=224, frames_per_batch=60):
        self.vid_folders = sorted(glob.glob("{}/*".format(path)))
        self.num_classes = len(self.vid_folders)

        self.vid_paths = [glob.glob("{}/*".format(vid_folder)) for vid_folder in self.vid_folders]
        self.vids_per_class = [len(glob.glob("{}/*".format(vid_folder))) for vid_folder in self.vid_folders]
        self.vid_dim = (vid_dim, vid_dim)
        self.frames_per_batch = frames_per_batch
        self.labels = torch.zeros(len(self.vid_paths), self.num_classes, dtype=torch.int8)
        self.label_names = [label.split('/')[-1] for label in self.vid_folders]

        start = 0
        for i, num_vids in enumerate(self.vids_per_class):
            for j in range(start, num_vids*(i+1)):
                self.labels[j][i] += 1
            start += num_vids

    def __len__(self):
        return len(self.vid_paths)

    def __getitem__(self, index):

        video_path = self.vid_paths[index].strip()
        frames = get_frames(video_path, self.frames_per_batch)
        for i, _ in enumerate(frames):
            frames[i] = cv2.resize(frames[i], self.vid_dim, interpolation=cv2.INTER_AREA)
        #need to convert frames to tensor, and each frame in their to tensors as well
        return frames, self.labels[index]
"""
dataset for loading youtube8m videos and generating a postive or negative pair
"""
class Youtube8m(Dataset):

    def __init__(self, path, vid_dim=224, frames_per_batch=60):
        self.vid_paths = glob.glob("{}/*".format(path))
        self.vid_dim = (vid_dim, vid_dim)
        self.frames_per_batch = frames_per_batch

    def __len__(self):
        return len(self.vid_paths)

    def __getitem__(self, index):
        label = np.random.randint(0, high=2)

        video_path = self.vid_paths[index].strip()
        frames = get_frames(video_path, self.frames_per_batch)
        for i, _ in enumerate(frames):
            frames[i] = cv2.resize(frames[i], self.vid_dim, interpolation=cv2.INTER_AREA)
            frames[i] /= 255.0 #normalize 0-255 -> 0-1

        #need to convert frames to tensor, and each frame in their to tensors as well
        if label:#positive pair
            return (frames, frames), label
        else:#negative pair, get another sample.
            #get randome video, loop incase it gets the same video
            neg_index = None
            while True:
                neg_index = np.random.randint(0, high=len(self.vid_paths))
                if neg_index != index:
                    break
            video_neg_path = self.vid_paths[neg_index].strip()
            neg_frames = get_frames(video_neg_path, self.frames_per_batch)
            for i, _ in enumerate(neg_frames):
                neg_frames[i] = cv2.resize(neg_frames[i], self.vid_dim, interpolation=cv2.INTER_AREA)
                neg_frames[i] /= 255.0
            return (frames, neg_frames), label
