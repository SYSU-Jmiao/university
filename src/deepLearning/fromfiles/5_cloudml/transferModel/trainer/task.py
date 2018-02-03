"""
cation of the mnist_mlp.py example on the Keras github repo.

This file is better suited to run on Cloud ML Engine's servers. It saves the
model for later use in predictions, uses pickled data from a relative data
source to avoid re-downloading the data every time, and handles some common
ML Engine parameters.
"""

from __future__ import print_function

import argparse
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.models import Sequential  
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical
from datetime import datetime  # for filename conventions
import os
import math  
from subprocess import check_call
import numpy as np


def get_data(data_location):
    print("getting data from: "+ data_location)
    tmp_path = '/tmp/data'
    os.mkdir(tmp_path)
    check_call(['gsutil', '-m', '-q', 'cp', '-r',data_location,tmp_path])


def save_bottleneck_features():
    batch_size = 60   
    epochs = 20  
    img_width, img_height = 224, 224  

    train_data_dir = '/tmp/data/3/train'  
    validation_data_dir = '/tmp/data/3/validation'  

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def copy_bottleneck_features_to_bucket(job_dir):
    print("save bottleneck features")
    check_call(['gsutil', 'cp','bottleneck_features_train.npy',job_dir])
    check_call(['gsutil', 'cp','bottleneck_features_validation.npy',job_dir])


def train(job_dir):
    print("train")
    save_bottleneck_features()
    copy_bottleneck_features_to_bucket(job_dir)

def train_top_model(job_dir):
    train_data_dir = '/tmp/data/3/train'  
    validation_data_dir = '/tmp/data/3/validation' 
    img_width, img_height = 224, 224  
    batch_size = 60
    epochs = 100 
    top_model_weights_path = 'bottleneck_fc_model.h5' 

 
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
    

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels),
                        callbacks = [
                            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
                            keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=10, write_graph=True, write_images=True)
                        ]
                       )

    model.save_weights(top_model_weights_path)
    check_call(['gsutil', 'cp',top_model_weights_path,job_dir])
    check_call(['gsutil', 'cp', '-r','./logs',job_dir])

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

# Create a function to allow for different training data and other options
def train_model(data_location='data/',
                job_dir='./job_dir', **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    get_data(data_location)
    train(job_dir)
    train_top_model(job_dir)


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data-location',
      help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
