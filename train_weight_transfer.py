# Code to train T3D model
import os
import numpy as np
import pandas as pd
import pickle
import time
import argparse
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, SGD
import keras.backend as K
import traceback

from models_src.T3D_keras import densenet161_3D_DropOut_wt
from data_loader.get_video import video_gen_wt

parser = argparse.ArgumentParser()
parser.add_argument("-train_csv_path", type=str, default="data/train_wt.csv", help="path to training csv file")
parser.add_argument("-valid_csv_path", type=str, default="data/test_wt.csv", help="path to valid csv file")
parser.add_argument("-video_resol", type=str, default="256x256", help="enter input resolution into network as WxH")
parser.add_argument("-frames_per_video", type=int, default=45, help="amount of frames to pull from each video to pass through to the network")
parser.add_argument("-batch_size", type=int, default=1, help="batch size of input data")
parser.add_argument("-epochs", type=int, default=200, help="number of epochs to run")
parser.add_argument("-model_save_path", type=str, default="saved_weights/", help="directory to save weights and models to")
parser.add_argument("-weight_load_path", type=str, default=None, help="path to pretrained weights to load")
options = parser.parse_args()

start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())

# there is a minimum number of frames that the network must have, values below 10 gives -- ValueError: Negative dimension size caused by subtracting 3 from 2 for 'conv3d_7/convolution'
# paper uses 224x224, but in that case also the above error occurs
FRAMES_PER_VIDEO = options.frames_per_video
FRAME_HEIGHT = int(options.video_resol.split("x")[1])
FRAME_WIDTH = int(options.video_resol.split("x")[0])
FRAME_CHANNEL = 3
BATCH_SIZE = options.batch_size
EPOCHS = options.epochs
MODEL_FILE_NAME = '{}/T3D_saved_model_wt_{}.h5'.format(options.model_save_path, start_time)
learning_rate = 1e-4
decay = 1e-6

def train():
    sample_input = np.empty([FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)

    # Read Dataset
    d_train = pd.read_csv(os.path.join(options.train_csv_path))
    d_valid = pd.read_csv(os.path.join(options.valid_csv_path))

    video_train_generator = video_gen_wt(
        d_train, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, batch_size=BATCH_SIZE)
    video_val_generator = video_gen_wt(
        d_valid, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, batch_size=BATCH_SIZE)

    # Get Model
    # model = densenet121_3D_DropOut(sample_input.shape, nb_classes)
    model = densenet161_3D_DropOut_wt(sample_input.shape)
    #model.summary()
    checkpoint = ModelCheckpoint('saved_weights/T3D_saved_model_weights_wt_{}.hdf5'.format(start_time), monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=30)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=20,
                                       verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-9)
    ten_board = TensorBoard(log_dir='tensorboard_logs/wt_{}'.format(start_time), write_images=True)

    callbacks_list = [checkpoint, reduceLROnPlat, earlyStop, ten_board]

    # compile model
    optim = Adam(lr=learning_rate, decay=decay)
    #optim = SGD(lr = 0.1, momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])

    if options.weight_load_path is not None:
        model.load_weights(options.weight_load_path)
        print('Weights loaded')

    # train model
    print('Training started....')

    train_steps = len(d_train)//BATCH_SIZE
    val_steps = len(d_valid)//BATCH_SIZE

    history = model.fit_generator(
        video_train_generator,
        steps_per_epoch=train_steps,
        epochs=EPOCHS,
        validation_data=video_val_generator,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callbacks_list,
        workers=1
    )
    model.load_weights('saved_weights/T3D_saved_model_weights_wt_{}.hdf5'.format(start_time))
    model.save(MODEL_FILE_NAME)
    with open("data/history_wt_{}.pkl".format(start_time), "wb") as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    try:
        train()
    except Exception as err:
        print('Error:', err)
        traceback.print_tb(err.__traceback__)
    finally:
        # Destroying the current TF graph to avoid clutter from old models / layers
        K.clear_session()
