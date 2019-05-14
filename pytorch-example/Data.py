import os, sys
import numpy as np
import cv2
from random import randint, choice
from torch.utils.data import Dataset, DataLoader
from math import ceil
import torch
import matplotlib.pyplot as plt
from torchvision import transforms


class UCF101Dataset(Dataset):

    def __init__(self, class_idxs_path, split_path, vid_root, clip_len=16, test=False, transform=None):

        """
        Args:
            split_path (string): Path to train or test split
            vid_root (string): Directory (root) of directories (classes) of directories (vid_id) of frames.
            aud_root (string): Directory (root) of directories (classes) of directories (aud_id) of audio tracks.
                default cluster paths:
                class_idxs_path = '/home/robert/Datasets/UCF_101/UCF101_Audio_Splits/classInd_audio.txt'
                split_path = '/home/robert/Datasets/UCF_101/UCF101_Audio_Splits/aud_trainlist01.txt'
                vid_root = '/home/robert/Datasets/UCF_101/UCF_101_Frames'
                aud_root = '/home/robert/Datasets/UCF_101/UCF_101_Audio'

            clip_len (int): Number of frames per sample, i.e. depth of Model input.
            aud_clip_len (float): Length of audio, in seconds, per "clip".
            test (bool): Testing vs. Training model.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.class_idxs_path = class_idxs_path
        self.split_path = split_path
        self.vid_root = vid_root
        self.clip_len = clip_len
        self.test = test
        self.transform = transform
        self.class_dict = self.read_class_ind_file()
        self.paths = self.read_split()



    def read_class_ind_file(self):
        class_dict = {}
        with open(self.class_idxs_path) as f:
            for line in f:
                class_value, class_key = line.strip().split()
                if class_key not in class_dict:
                    class_dict[class_key] = []
                class_dict[class_key] = class_value #.append(line.strip())

        return class_dict


    def read_split(self):
        class_dict = self.class_dict
        paths = []
        with open(self.split_path) as f:
            for line in f:
                x1 = line.strip().split()[0]
                class_name, _vid_id = x1.split('/')
                vid_id = _vid_id[:-4]
                paths.append((class_name, vid_id, class_dict[class_name]))

        return paths


    def random_index(self, vid_id):
        path = os.path.join(self.vid_root, vid_id)
        frames = [int(frame[:-4]) for frame in os.listdir(path) if not frame.startswith('.')]
        num_frames = max(frames)
        rand_idx = randint(1, num_frames - self.clip_len)

        return rand_idx, num_frames


    def vid_clip_builder(self, vid_id, rand_idx):
        vid_path = os.path.join(self.vid_root, vid_id)
        vid_clip = []
        for i in range(self.clip_len):
            frame_path = os.path.join(vid_path, str(rand_idx + i) + '.jpg')
            frame = cv2.imread(frame_path)
            if frame is None:
                print("Could not load file %s" % (frame_path))
                sys.exit()
            frame_crop = frame[:, 40:280, :]
            frame_resize = cv2.resize(frame_crop, (112, 112))
            # image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ; Check syncnet
            vid_clip.append(frame_resize)
        vid_clip = np.array(vid_clip).transpose([3, 0, 1, 2]).astype(dtype='Float32')

        return vid_clip



    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        class_name, vid_id, class_idx = self.paths[idx]
        rand_idx, frame_cnt = self.random_index(vid_id)
        vid_clip = self.vid_clip_builder(vid_id, rand_idx)
        # class_idx = torch.autograd.Variable(torch.from_numpy(class_idx.astype(float)).float())
        class_idx = int(class_idx) - 1

        return vid_clip, class_idx




if __name__ == '__main__':
    class_idxs_path = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/UCF_101/UCF10_Splits/classInd.txt'
    split_path = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/UCF_101/UCF10_Splits/trainlist01.txt'
    vid_root = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/UCF_101/UCF10_Frames'
    dataset = UCF101Dataset(class_idxs_path, split_path, vid_root)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print('length: ', len(dataloader))
    for i in range(30):
        vid, class_idx = next(iter(dataloader))
        print('vid shape: ', vid.shape)
        print('class idxs: ', class_idx)
        print('='*100)
#     print('Batch shape:', imgs.numpy().shape)
#     plt.imshow(imgs.numpy()[0, 0, :, :, :])
#     plt.show()
#     plt.imshow(imgs.numpy()[0, 15, :, :, :])
#     plt.show()
