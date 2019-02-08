from torch.utils.data import Dataset
from torchvision import transforms, utils
import torch
import cv2
import numpy as np
import glob


"""
takes in path to a video and number of consecutive frames to get from video, and returns a random section of the video split into frames of length num_frames
if num_frames is 0, returns all frames in video
"""
def get_frames(video_path, num_frames):
    video = cv2.VideoCapture(video_path)
    frames = []
    #print("now reading from {}".format(video_path))
    ret, frame = video.read()
    while ret:
        frames.append(frame)
        ret, frame = video.read()

    video.release()
    #get all frames from video, concatenate so its just one numpy array instead of multiply numpy arrays in a numpy array
    #frames dimesions is (num_frames, height, width, channels)
    frames = np.concatenate(np.expand_dims(frames, axis=0))
    #if num_frames is 0, return all frames from video:
    if num_frames == 0:
        return frames
    #check to make sure frames can be split properly, if not remove last frame to make it even
    if frames.shape[0] % num_frames != 0:
        frames = frames[0:frames.shape[0] - 1]
    #split frames into batches of num_frames
    leftover = 0
    if frames.shape[0] % num_frames:
        leftover = 1
    num_splits = frames.shape[0] // num_frames + leftover
    frames = [np.concatenate(np.expand_dims(frames[i*num_frames : min((i + 1) * num_frames, frames.shape[0])], axis=0)) for i in range(num_splits)]
    #randomly select consecutive num_frames batch from frames
    return frames[np.random.choice(len(frames))]

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

        self.vid_paths = []
        for vid_folder in self.vid_folders:
            self.vid_paths.extend(glob.glob("{}/*".format(vid_folder)))
        self.vids_per_class = [len(glob.glob("{}/*".format(vid_folder))) for vid_folder in self.vid_folders]
        self.vid_dim = (vid_dim, vid_dim)
        self.frames_per_batch = frames_per_batch
        self.labels = torch.zeros(len(self.vid_paths), dtype=torch.int32)
        self.label_names = [label.split('/')[-1] for label in self.vid_folders]

        start = 0
        end = 0
        #set labels, from 0 to num_classes - 1
        for i, num_vids in enumerate(self.vids_per_class):
            end += num_vids
            while start < end:
                self.labels[start] = i
                start += 1

    def __len__(self):
        return len(self.vid_paths)

    def __getitem__(self, index):

        video_path = self.vid_paths[index].strip()
        frames = get_frames(video_path, self.frames_per_batch)
        resized_frames = []

        for i, _ in enumerate(frames):
            resized_frames.append(cv2.resize(frames[i], self.vid_dim, interpolation=cv2.INTER_AREA))

        frames = np.rollaxis(np.concatenate(np.expand_dims(resized_frames, axis=0)), 3, 1)
        frames = frames.astype('float32') / 255.0 #normalize 0-255 to 0-1

        return torch.from_numpy(frames), self.labels[index]

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
        label = np.random.randint(2)

        video_path = self.vid_paths[index].strip()
        frames = get_frames(video_path, self.frames_per_batch)
        resized_frames = []

        for i, _ in enumerate(frames):
            resized_frames.append(cv2.resize(frames[i], self.vid_dim, interpolation=cv2.INTER_AREA))

        frames = np.rollaxis(np.concatenate(np.expand_dims(resized_frames, axis=0)), 3, 1)
        frames = frames.astype('float32') / 255.0 #normalize 0-255 to 0-1

        if label:#positive pair
            return torch.from_numpy(frames), torch.tensor(label)
        else:#negative pair, get another sample.
            #get randome video, loop incase it gets the same video
            neg_index = None
            while True:
                neg_index = np.random.randint(0, high=len(self.vid_paths))
                if neg_index != index:
                    break
            video_neg_path = self.vid_paths[neg_index].strip()

            neg_frames = get_frames(video_path, self.frames_per_batch)
            resized_frames = []

            for i, _ in enumerate(neg_frames):
                resized_frames.append(cv2.resize(neg_frames[i], self.vid_dim, interpolation=cv2.INTER_AREA))

            neg_frames = np.rollaxis(np.concatenate(np.expand_dims(resized_frames, axis=0)), 3, 1)
            neg_frames = frames.astype('float32') / 255.0 #normalize 0-255 to 0-1
            neg_frames = np.concatenate((frames, neg_frames), axis=0)
            return torch.from_numpy(neg_frames), torch.tensor(label)
