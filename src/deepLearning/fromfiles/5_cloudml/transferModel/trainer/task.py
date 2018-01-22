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


def save_bottlebeck_features():
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


def train():
    print("train")
    save_bottlebeck_features()

def copy_meta_to_bucket():
    print("copy meta")


# Create a function to allow for different training data and other options
def train_model(data_location='data/',
                job_dir='./job_dir', **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    get_data(data_location)
    train()
    copy_meta_to_bucket()



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
