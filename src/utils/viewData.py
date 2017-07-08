import os,random
import numpy as np
import matplotlib.pyplot as plt
import cPickle, random, sys


Xd = cPickle.load(open("RML2016.10a_dict.dat",'rb'))
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

test_value = X_train[0]

t = np.arange(0,128,1)
s = test_value[0]
u = test_value[1]

plt.plot(t,s)

plt.xlabel('sample')
plt.ylabel('voltage (mV)')
plt.grid(True)

plt.figure(1)
plt.subplot(211)
plt.title('I channel')
plt.plot(t,s)

plt.subplot(212)
plt.title('Q channel')
plt.plot(t,u)

plt.show()