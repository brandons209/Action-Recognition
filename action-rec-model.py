# general imports:
import argparse

# keras model imports
from keras.layers import Activation
from keras.layers import Conv3D
from keras.layers import MaxPooling3D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten

from keras.models import Sequential

# optimizers and loss functions
from keras.layers.advanced_activations import LeakyReLU
import keras.optimizers as opt
from keras.utils import np_utils
