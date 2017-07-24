import os
import random

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_FLAGS"] = "device=gpu%d" % (0)

import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle
import random
import sys
import keras
import missinglink

from IPython.core.interactiveshell import InteractiveShell


# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("/opt/RML2016.10a_dict.dat", 'rb'))
snrs, mods = map(lambda j: sorted(
    list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)


# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(
    range(0, int(n_examples)), size=int(n_train), replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
X_train = np.expand_dims(X[train_idx], 3)
X_test = np.expand_dims(X[test_idx], 3)


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))


in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods


######
from keras import applications
model = applications.VGG16(include_top=False, weights='imagenet')


dr = 0.5  # dropout rate (%)

model = models.Sequential()

model.add(Flatten(input_shape=in_shp))
model.add(Dense(256, kernel_initializer="he_normal",
                activation="relu", name="dense1"))
model.add(Dropout(dr))
model.add(Dense(11, kernel_initializer="he_normal", name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(mods)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# Set up some params
nb_epoch = 50     # number of epochs to train on
batch_size = 16  # training batch size


# Generate

from scipy import interpolate, signal


def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0 / overSampleFactor)
    sNew = f(tNew)
    return sNew


def cwt(x):
    index = random.randint(1, 100)

    vector = x[0]
    sNew = getOverSampledSignal(vector, 8)
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.draw()
    plt.savefig('./backups/' + str(index) + '_I' + '.png', dpi=100)
    plt.close()

    vector = x[1]
    sNew = getOverSampledSignal(vector, 8)
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.draw()
    plt.savefig('./backups/' + str(index) + '_Q' + '.png', dpi=100)
    plt.close()

    return


def preprocessor(x):
    image = cwt(x)
    return x


train_datagen = ImageDataGenerator(preprocessing_function=preprocessor)
train_generator = train_datagen.flow(
    X_train, Y_train, batch_size, save_to_dir="generated")

validate_datagen = ImageDataGenerator(preprocessing_function=preprocessor)
validate_generator = train_datagen.flow(X_test, Y_test, batch_size)

# perform training ...
missinglink_callback = missinglink.KerasCallback(
    owner_id="73b7dbec-273d-c6b7-776d-55812449a4e4", project_token="WxqnIeHhwiLIFejy")
missinglink_callback.set_properties(
    display_name='Base experiment', description='Initial base mode for experiment')
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit_generator(
    train_generator,
    steps_per_epoch=X_train.shape[0] / batch_size,
    validation_data=validate_generator,
    validation_steps=X_test.shape[0] / batch_size,
    epochs=nb_epoch,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=0, mode='auto'),
        keras.callbacks.TensorBoard(
            log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)


score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score
