import os,random
import numpy as np
import matplotlib.pyplot as plt
import cPickle, random, sys
from scipy import interpolate
from scipy import signal

DB_LOCATION = "RML2016.10a_dict.dat"

def getSamples(dbLocation):
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
    return X_train


def cwtSignal(index,list):
    test_value = list[0]
    plotCWT(test_value[0])
    plotCWT(test_value[1])
    

def plotCWT(mySignal):
    t = np.arange(0,128,1)
    f = interpolate.InterpolatedUnivariateSpline(t, mySignal)
    tNew = np.arange(0.0,128.0,0.125)
    sNew = f(tNew)
    widths = np.arange(1, 256)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, 256], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    print cwtmatr.shape       
    plt.show()


samplesList = getSamples(DB_LOCATION)

cwtSignal(0,samplesList)
cwtSignal(1,samplesList)
cwtSignal(2,samplesList)