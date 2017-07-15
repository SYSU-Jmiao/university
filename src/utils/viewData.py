import os, random, cPickle, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

from dbManager import initDb
from signalUtils import getSignalWithLabelGenerator
from signalUtils import crossSpectrumDensity, spectogram, periodGram, cwt

NUMBER_OF_SAMPLES_PER_METHOD = 6
NUMBER_OF_ROWS = 2
NUMBER_OF_COLUMNS = 3

generator = getSignalWithLabelGenerator(initDb)

for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD):
    sample, label = generator(x)
    plt.subplot(3, 2, x+1)
    plt.tight_layout()
    plt.title(label)
    crossSpectrumDensity(sample)

plt.show();

for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD-1, 2):
    sample, label = generator(x)
    plt.subplot(3, 2, x+1)
    plt.tight_layout()
    spectogram(sample[0])
    plt.title( " I " + label)
    plt.subplot(3, 2, x+2)
    plt.tight_layout()
    spectogram(sample[1])
    plt.title( " Q " + label)

plt.show();

for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD-1, 2):
    sample, label = generator(x)
    plt.subplot(3, 2, x+1)
    plt.tight_layout()
    periodGram(sample[0])
    plt.title( " I " + label)
    plt.subplot(3, 2, x+2)
    plt.tight_layout()
    periodGram(sample[1])
    plt.title( " Q " + label)

plt.show();

for x in range(0, NUMBER_OF_SAMPLES_PER_METHOD-1, 2):
    sample, label = generator(x)
    plt.subplot(3, 2, x+1)
    plt.tight_layout()
    cwt(sample[0])
    plt.title( " I " + label)
    plt.subplot(3, 2, x+2)
    plt.tight_layout()
    cwt(sample[1])
    plt.title( " Q " + label)

plt.show();