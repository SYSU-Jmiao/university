#!/usr/bin/env python
import cPickle
import matplotlib     
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import os


def create_image(index,value,label,snr):
    print index
    print value
    print label
    print snr

def create_folder(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)

BASE_PATH="/tmp/data_test"
data = cPickle.load( open( "../data_dict.dat", "rb" ) )

create_folder(BASE_PATH)
for p in data:
    label = p[0]
    snr = str(p[1])
    print "label:"+label+ ", SNR:"+ snr
    create_folder(os.path.join(BASE_PATH,label))
    [create_image(index, value,label,snr) for index, value in enumerate(data[p])]