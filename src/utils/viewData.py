import os, random, cPickle, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal
from dbManager import initDb

def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0/overSampleFactor)
    sNew = f(tNew)
    return sNew

def cwt(x):
    sNew = getOverSampledSignal(x, 8)
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.plot()

def periodGram(x):
    fs = 10e6
    f, Pxx_den = signal.periodogram(getOverSampledSignal(x,8), fs)
    plt.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.plot()

def spectogram(x):
    fs = 10e6
    f, t, Sxx = signal.spectrogram(getOverSampledSignal(x, 8), fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.plot()

def crossSpectrumDensity(sample):
    fs = 10e6
    x, y = sample[0], sample[1]
    f, Pxy = signal.csd(getOverSampledSignal(x, 8),getOverSampledSignal(y, 8), fs, nperseg=1024)
    plt.semilogy(f, np.abs(Pxy))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('CSD [V**2/Hz]')
    plt.plot()

def getSignalWithLabelGenerator(initDb):
    signals, labels, mods = initDb()
    def getSignalWithLabel(x):
        signal = signals[x]
        label = mods[np.where(labels[x] == 1.0)[0][0]]
        return  signal, label
    return getSignalWithLabel


generator = getSignalWithLabelGenerator(initDb)
numberOfSamples = 6
for x in range(0, numberOfSamples):
    sample, label = generator(x)
    plt.subplot(3, 2, x+1)
    plt.tight_layout()
    plt.title(label)
    crossSpectrumDensity(sample)

plt.show();

numberOfSamples = 6
for x in range(0, numberOfSamples-1, 2):
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

numberOfSamples = 6
for x in range(0, numberOfSamples-1, 2):
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

numberOfSamples = 6
for x in range(0, numberOfSamples-1, 2):
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