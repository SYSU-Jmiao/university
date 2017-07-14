import os, random, cPickle, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate, signal

DB_LOCATION = "RML2016.10a_dict.dat"

def initDataBase(dbLocation):
    Xd = cPickle.load(open(dbLocation,'rb'))
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
    X = np.vstack(X)
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
    return X_train,Y_train,mods

def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    print 'Original size = ', sampleSize,', New size = ', sampleSize*overSampleFactor
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0/overSampleFactor)
    sNew = f(tNew)
    return sNew


def cwtSignal(index,list):
    test_value = list[index]
    plotCWT(test_value[0])
    plotCWT(test_value[1])
    

def plotCWT(mySignal):
    sNew = getOverSampledSignal(mySignal, 8)
    print sNew.shape
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    print cwtmatr.shape
    plt.show()

def periodGram(index,list):
    test_value = list[index]
    plotPeriodGram(test_value[0])
    plotPeriodGram(test_value[1])

def plotPeriodGram(x):
    fs = 10e6
    f, Pxx_den = signal.periodogram(getOverSampledSignal(x,8), fs)
    plt.semilogy(f, Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD [V**2/Hz]')
    plt.show()


def spectogram(index,list):
    test_value = list[index]
    plotspectogram(test_value[0])
    plotspectogram(test_value[1])

def plotspectogram(x):
    fs = 10e6
    f, t, Sxx = signal.spectrogram(getOverSampledSignal(x, 8), fs)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def crossSpectrumDensity(sample, label):
    fs = 10e6
    x, y = sample[0], sample[1]
    f, Pxy = signal.csd(getOverSampledSignal(x, 8),getOverSampledSignal(y, 8), fs, nperseg=1024)
    plt.semilogy(f, np.abs(Pxy))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('CSD [V**2/Hz]')
    plt.title(label)
    plt.show()

def getSignalWithLabelGenerator(initDb,dbName):
    signals, labels, mods = initDb(dbName)
    def getSignalWithLabel(x):
        signal = signals[x]
        label = mods[np.where(labels[x] == 1.0)[0][0]]
        return  signal, label
    return getSignalWithLabel


generator = getSignalWithLabelGenerator(initDataBase, DB_LOCATION)

for x in range(0, 20):
    sample, label = generator(x)
    crossSpectrumDensity(sample, label)
