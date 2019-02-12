import argparse
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from data_loader.get_video import video_gen

parser = argparse.ArgumentParser()
parser.add_argument("-test_csv_path", type=str, default="data/test.csv", help="path to testing csv file")
parser.add_argument("-video_resol", type=str, default="256x256", help="enter input resolution into network as WxH")
parser.add_argument("-frames_per_video", type=int, default=45, help="amount of frames to pull from each video to pass through to the network")
parser.add_argument("-batch_size", type=int, default=1, help="batch size of input data")
parser.add_argument("-model_path", type=str, default=None, help="path to model to load")
options = parser.parse_args()


FRAMES_PER_VIDEO = options.frames_per_video
FRAME_HEIGHT = int(options.video_resol.split("x")[1])
FRAME_WIDTH = int(options.video_resol.split("x")[0])
FRAME_CHANNEL = 3
BATCH_SIZE = options.batch_size

model = load_model(options.model_path)

d_test = pd.read_csv(os.path.join(options.test_csv_path))
nb_classes = len(set(d_test['class']))

video_test_generator = video_gen(d_test, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)
test_steps = len(d_test)//BATCH_SIZE

metrics = model.evaluate_generator(video_test_generator, steps=test_steps, verbose=1)

for i, metric in enumerate(model.metrics_names):
    print("{}: {:.3f}".format(metric, metrics[i]))
