
# coding: utf-8

# In[32]:


import pickle
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from PIL import Image



# In[152]:


# Load the dataset ...
Xd = pickle.load(open("../RML2016.10a_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)

X_snr = map(lambda x: lbl[x][1], range(0,n_examples))
X_mod = map(lambda x: lbl[x][0], range(0,n_examples))


# In[153]:


n_examples = X.shape[0]
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(0,n_examples)))


# In[154]:


print "X props"
print X.shape, list(X.shape[1:])
print "Y props"
print Y.shape , list(Y.shape[1:])
print "classes"
print mods
print "snrs"
print len(X_snr)
print "mods"
print len(X_mod)


# In[155]:


def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0 / overSampleFactor)
    sNew = f(tNew)
    return sNew


# In[156]:


def cwt(name,sample):
    iNew = getOverSampledSignal(sample[0], 8)
    qNew = getOverSampledSignal(sample[1], 8)
    width = 16
    widths = np.arange(1, width)
    iCwtmatr = signal.cwt(iNew, signal.ricker, widths)
    qCwtmatr = signal.cwt(qNew, signal.ricker, widths)

    f, axarr = plt.subplots(2, sharex=True)

    axarr[0].imshow(iCwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(iCwtmatr).max(), vmin=-abs(iCwtmatr).max())
    axarr[1].imshow(qCwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(qCwtmatr).max(), vmin=-abs(qCwtmatr).max())
    axarr[0].axis('off')
    axarr[1].axis('off')

    plt.savefig(name, format='png', bbox_inches='tight', pad_inches=0)

    plt.close()


# In[179]:


def getPath(index, snrs, mods):
  path = "./data/" + str(mods[index]) + "/" + str(index)+ "_" + str(mods[index]) + "_" + str(snrs[index]) + ".png"
  print path
  return path


# In[180]:


import os
def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


# In[181]:


mkdir("./data")
map(lambda x: mkdir("./data/"+x), mods)


# In[183]:


for index, item in enumerate(X):
    path = getPath(index, X_snr, X_mod)
    cwt(path, item)
    if index == 10000:
      break


# In[ ]:
