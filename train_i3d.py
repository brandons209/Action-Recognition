import time
import argparse
import numpy as np
import keras.backend as K
import traceback
import pickle
import pandas as pd
import os
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam, SGD
from models_src.i3d_inception import Inception_Inflated3d
from data_loader.get_video import video_gen

parser = argparse.ArgumentParser()
parser.add_argument("-train_csv_path", type=str, default="data/train.csv", help="path to training csv file")
parser.add_argument("-valid_csv_path", type=str, default="data/test.csv", help="path to valid csv file")
parser.add_argument("-video_resol", type=str, default="224x224", help="enter input resolution into network as WxH")
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
MODEL_FILE_NAME = '{}/I3D_saved_model_{}.h5'.format(options.model_save_path, start_time)
learning_rate = 1e-4
decay = 1e-6

def train():
    sample_input = np.empty([FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL], dtype=np.uint8)
    # Read Dataset
    d_train = pd.read_csv(os.path.join(options.train_csv_path))
    d_valid = pd.read_csv(os.path.join(options.valid_csv_path))
    # Split data into random training and validation sets
    nb_classes = len(set(d_train['class']))

    video_train_generator = video_gen(
        d_train, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)
    video_val_generator = video_gen(
        d_valid, FRAMES_PER_VIDEO, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, nb_classes, batch_size=BATCH_SIZE)

    # Get Model
    model = Inception_Inflated3d(include_top=False, input_shape=sample_input.shape, weights="rgb_kinetics_only")
    classifier = model.output
    classifier = Flatten()(classifier)
    classifier = Dense(nb_classes, activation='softmax')(classifier)
    model = Model(inputs=model.input, outputs=classifier)
    #model.summary()
    checkpoint = ModelCheckpoint('saved_weights/I3D_saved_model_weights_{}.hdf5'.format(start_time), monitor='val_loss',
                                 verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=25)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                       patience=20,
                                       verbose=1, mode='min', min_delta=0.0001, cooldown=2, min_lr=1e-9)
    ten_board = TensorBoard(log_dir='tensorboard_logs/i3d_{}'.format(start_time), write_images=True)

    callbacks_list = [checkpoint, reduceLROnPlat, earlyStop, ten_board]

    # compile model
    optim = Adam(lr=learning_rate, decay=decay)
    #optim = SGD(lr = 0.1, momentum=0.9, decay=1e-4, nesterov=True)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

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
    model.load_weights('saved_weights/I3D_saved_model_weights_{}.hdf5'.format(start_time))
    model.save(MODEL_FILE_NAME)
    with open("data/history_i3d_{}.pkl".format(start_time), "wb") as f:
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
