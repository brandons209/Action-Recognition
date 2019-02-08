import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
import random

ROOT_PATH = ''

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

    rnd_idx = random.randint(5,len(frames)-5)
    rnd_frame = frames[rnd_idx]
    rnd_frame = cv2.resize(rnd_frame,(224,224), interpolation=cv2.INTER_AREA) #Needed for Densenet121-2d
    #print("frame length: ", len(frames))
    #if the amount of frames is less than the frames per video, set step to one
    step = len(frames)//fpv
    if step == 0:
        step = 1
    #gets which section of frames to select
    frame_select = np.random.choice(np.arange(step))
    avg_frames = frames[(frame_select*fpv):(frame_select*fpv + fpv)]
    #print("avg frame length: ", len(avg_frames))
    #avg_frames = avg_frames[:fpv]
    avg_resized_frames = []
    for af in avg_frames:
        rsz_f = cv2.resize(af, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        rsz_f = rsz_f / 255.0
        avg_resized_frames.append(rsz_f)
    #print("shape: ", np.asarray(avg_resized_frames).shape)
    return np.asarray(rnd_frame)/255.0, np.asarray(avg_resized_frames)


def get_video_and_label(index, data, frames_per_video, frame_height, frame_width, use_class=True):
    # Read clip and appropiately send the sports' class
    frame, clip = get_video_frames(os.path.join(
        ROOT_PATH, data['path'].values[index].strip()), frames_per_video, frame_height, frame_width)
    if use_class:
        action = data['class'].values[index]
    else:
        action = None

    frame = np.expand_dims(frame, axis=0)
    clip = np.expand_dims(clip, axis=0)

    # print('Frame shape',frame.shape)
    # print('Clip shape',clip.shape)
    return frame, clip, action


def video_gen(data, frames_per_video, frame_height, frame_width, channels, num_classes, batch_size=4):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            clip = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)

            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                # get frames and its corresponding color for an traffic light
                _, single_clip, action = get_video_and_label(i, data, frames_per_video, frame_height, frame_width)
                 # Appending them to existing batch
                clip = np.append(clip, single_clip, axis=0)
                y_train = np.append(y_train, [action])

            y_train = to_categorical(y_train, num_classes=num_classes)

            yield (clip, y_train)

def video_gen_wt(data, frames_per_video, frame_height, frame_width, channels, batch_size=4):
    while True:
        # Randomize the indices to make an array
        indices_arr = np.random.permutation(data.count()[0])
        for batch in range(0, len(indices_arr), batch_size):
            # slice out the current batch according to batch-size
            current_batch = indices_arr[batch:(batch + batch_size)]

            # initializing the arrays, x_train and y_train
            clip = np.empty([0, frames_per_video, frame_height, frame_width, channels], dtype=np.float32)
            frame = np.empty([0, 224, 224, 3], dtype=np.float32)

            label = np.random.randint(2);
            y_train = np.empty([0], dtype=np.int32)

            for i in current_batch:
                if label == 0:
                    #generate index for negative sample, and makes sure it is not the current index.
                    neg_index = np.random.randint(len(data['path']))
                    while neg_index == i:
                        neg_index = np.random.randint(len(data['path']))
                    #get clip from current index, and frame from negative index
                    _, single_clip, _ = get_video_and_label(
                        i, data, frames_per_video, frame_height, frame_width, use_class=False)
                    single_frame, _ , _ = get_video_and_label(
                            neg_index, data, frames_per_video, frame_height, frame_width, use_class=False)
                else:
                    single_frame, single_clip, _ = get_video_and_label(
                        i, data, frames_per_video, frame_height, frame_width, use_class=False)

                # Appending them to existing batch
                frame = np.append(frame, single_frame, axis=0)
                clip = np.append(clip, single_clip, axis=0)

                y_train = np.append(y_train, [label])

            yield ([frame, clip], y_train)
