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
from datetime import datetime  # for filename conventions
import sys
from subprocess import call


def get_data(data_location):
    print("getting data from: "+ data_location)
    destination='./data'
    call(['mkdir' ,destination])
    call(['gsutil' ,'-m','cp','-r',data_location,destination])
    
    call('ls ./data/')


def train():
    print("demo")

def copy_meta_to_bucket():
    print("demo")


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
