import cPickle
import io
import os
import random
import sys

import keras
import keras.models as models
import matplotlib.pyplot as plt
import missinglink
import numpy as np
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell
######
from keras import applications
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.noise import GaussianNoise
from keras.optimizers import adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.utils import np_utils
from PIL import Image
from scipy import interpolate, signal

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_FLAGS"] = "device=gpu%d" % (0)

GENERATED_PATH = './generated/'
if not os.path.exists(GENERATED_PATH):
    os.mkdir(GENERATED_PATH)


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


in_shp = [224, 224, 3]
print X_train.shape, in_shp
classes = mods


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
batch_size = 200  # training batch size


# Generate

def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0 / overSampleFactor)
    sNew = f(tNew)
    return sNew


def cwt(x):
    sNew = getOverSampledSignal(x, 8)
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.draw()
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def preprocessor(x, name):
    i, q = x[0], x[1]
    vgg16ImageSize = (224, 224)
    finalImage = np.empty((224, 224, 3), dtype=np.uint8)

    iImg = cwt(i)
    qImg = cwt(q)

    iImg = iImg.resize(vgg16ImageSize)
    grayI = iImg.convert('L')
    qImg = qImg.resize(vgg16ImageSize)
    grayQ = qImg.convert('L')

    h, w = qImg.size
    finalImage[:, :, 0] = grayI
    finalImage[:, :, 1] = grayQ
    # Image.fromarray(finalImage).save(name + ".png")
    return finalImage


def train_generator(x, y, prefix):
    index = 0
    while index < x.shape[0]:
        labels = y[index:(index + batch_size)]
        samples = np.empty((batch_size, 224, 224, 3), dtype=np.uint8)
        originalSamples = x[index:(index + batch_size)]
        xIndex = 0
        while xIndex < batch_size:
            name = GENERATED_PATH + prefix + "_" + str(index + xIndex)
            samples[xIndex, :, :, :] = preprocessor(
                originalSamples[xIndex], name)
            xIndex = xIndex + 1
        yield(samples, labels)
        index = index + batch_size


#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'

history = model.fit_generator(
    train_generator(X_train, Y_train, "train"),
    max_queue_size=50,
    # workers=4,
    use_multiprocessing=True,
    steps_per_epoch=(X_train.shape[0] / batch_size),
    epochs=nb_epoch,
    validation_data=train_generator(X_test, Y_test, "validate"),
    validation_steps=X_test.shape[0] / batch_size,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, verbose=0, mode='auto'),
        keras.callbacks.TensorBoard(
            log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)


# score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print "**** CALCULATING SCORE*****"
score = model.evaluate_generator(
    train_generator(X_test, Y_test, "evaluate"), steps=(X_test.shape[0] / batch_size))
print score
