import os
import cv2
import numpy as np
from random import randint
import sys

root_path = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/UCF_101/'
frame_dir = 'UCF10_Frames/v_ApplyEyeMakeup_g01_c01'


def random_index(root, frame_dir, clip_len=16):
    path = os.path.join(root, frame_dir)
    frames = [int(frame[:-4]) for frame in os.listdir(path) if not frame.startswith('.')]
    num_frames = max(frames)
    rand_idx = randint(1, num_frames - clip_len)

    return rand_idx, num_frames


def vid_clip_builder(root, frame_dir, rand_idx, clip_len=16):
    vid_path = os.path.join(root, frame_dir)
    vid_clip = []
    for i in range(clip_len):
        frame_path = os.path.join(vid_path, str(rand_idx + i) + '.jpg')
        frame = cv2.imread(frame_path)
        # cv2.imshow('Orig Frame %d' % (rand_idx + i), frame)
        # cv2.waitKey(1500)
        if frame is None:
            print("Could not load file %s" % (frame_path))
        frame_crop = frame[:, 40:280, :]
        frame_resize = cv2.resize(frame_crop, (112, 112))
        # cv2.imshow('Resize Frame %d' % (rand_idx + i), frame_resize)
        # cv2.waitKey(1500)
        # cv2.destroyAllWindows()
        # image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) ; Check syncnet
        vid_clip.append(frame_resize)
    vid_clip = np.array(vid_clip).transpose([3, 0, 1, 2]).astype(dtype='Float32')

    return vid_clip

# rand_idx, num_frames = random_index(root_path, frame_dir)
# print(rand_idx)
# print(num_frames)
#
# vid_clip = vid_clip_builder(root_path, frame_dir, rand_idx)
# print(vid_clip.shape)


def mean_and_std_np(dataset):
    mean = np.zeros((3,1))
    std = np.zeros((3,1))
    print('==> Computing mean and std...')
    for img in dataset:
        scaled_img = np.array(img[0])/255
        mean_tmp, std_tmp = cv2.meanStdDev(scaled_img)
        mean += mean_tmp
        std += std_tmp
    mean = mean/len(dataset)
    std = std/len(dataset)

    return mean, std

# t0 = time.time()
# mean, std = mean_and_std_np(trainset)
# tF = time.time() - t0
# print('Time: ', tF)
# print('mean: ', mean)
# print('std: ', std)


# N = 0
# running_total = np.zeros((32, 32, 3))
#
# for img in trainset:
#     running_total += np.array(img[0])/255
#     N += 1
#
# # for img in testset:
# #     running_total += np.array(img[0])/255
# #     N += 1
#
# final = running_total/N



trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=None)

img, label = trainset[0]
print('img type:', type(img))
print('img hxw or wxh: ', img.size)
print('img shape: ', np.array(img).shape)

print('label type:', type(label))
print('label: ', label)

print(50*'=')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=1)

for img, label in trainloader:
    print('img type:', type(img))
    print('img hxw or wxh: ', img.size)
    print('img shape: ', np.array(img).shape)

    print('label type:', type(label))
    print('label: ', label)
    break

# frames = [os.path.join(path, item) for item in os.listdir(path) if not item.startswith('.')]
# frames.sort()
# print(frames)
#
#
# running_total = np.zeros((240, 320, 3))
# N = 0
# t0 = time.time()
# for frame in frames:
#     img = cv2.imread(frame)/255
#     running_total += img
#     N += 1
# tF = time.time()
# print('len frames: ', len(frames))
# print('N = ', N)
# print(running_total/N)
# t = tF - t0
# print('time: ', t)
#
#
# path1 = '/Users/robertbrowning/Desktop/UCF/CRCV/Data/UCF_101/UCF10_Frames'
# x = [item for item in os.listdir(path1) if not item.startswith('.')]
# print('Total dirs: ', len(x))
# print('Total Time: ', t*len(x)/60)
# print('Total files: ', 164*len(x))

x1 = [1, 2, 3]
y1 = [3, 4, 5]
x1 = np.array(x1)
y1 = np.array(y1)
x2 = [1, 2, 3]
y2 = [3, 4, 5]
x2 = np.array(x2)
y2 = np.array(y2)

x2, y2 += x1, y1
print(x+y)