import os,random

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_FLAGS"]  = "device=gpu%d"%(0)

import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import cPickle, random, sys, keras
import missinglink

from IPython.core.interactiveshell import InteractiveShell



# Load the dataset ...
#  You will need to seperately download or generate this file
Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)



# Partition the data
#  into training and test sets of the form we can train/test on 
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,int(n_examples)), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))



in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods



# Build VT-CNN2 Neural Net model using Keras primitives -- 
#  - Reshape [N,2,128] to [N,1,2,128] on input
#  - Pass through 2 2DConv/ReLu layers
#  - Pass through 2 Dense layers (ReLu and Softmax)
#  - Perform categorical cross entropy optimization

dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp+[1]))
model.add(ZeroPadding2D((0,2)))
model.add(Conv2D(256,(1,3),kernel_initializer="glorot_uniform", name="conv1", activation="relu", padding="valid"))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80,(2,3),padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(256, kernel_initializer="he_normal", activation="relu", name="dense1"))
model.add(Dropout(dr))
model.add(Dense(11, kernel_initializer="he_normal", name="dense2"))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()


# Set up some params 
nb_epoch = 100     # number of epochs to train on
batch_size = 200  # training batch size


# Generate

def preprocessor(x):
    return x

train_datagen = ImageDataGenerator(preprocessing_function=preprocessor)
train_generator = train_datagen.flow(np.expand_dims(X_train, 3), Y_train, batch_size)

validate_datagen = ImageDataGenerator(preprocessing_function=preprocessor)
validate_generator = train_datagen.flow(np.expand_dims(X_test, 3), Y_test, batch_size)

# perform training ...
missinglink_callback = missinglink.KerasCallback(owner_id="73b7dbec-273d-c6b7-776d-55812449a4e4", project_token="WxqnIeHhwiLIFejy")
missinglink_callback.set_properties(display_name='Base experiment', description='Initial base mode for experiment')
#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history = model.fit_generator(
    train_generator,    
    steps_per_epoch=X_train.shape[0]/batch_size,
    validation_data=validate_generator,
    validation_steps=X_test.shape[0]/batch_size,
    epochs=nb_epoch,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
        keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)


score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print score