#!/usr/bin/env python
import cPickle
import matplotlib     
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

data = cPickle.load( open( "../data_dict.dat", "rb" ) )

for p in data:
    label = p[0]
    snr = str(p[1])
    print "label:"+label+ ", SNR:"+ snr