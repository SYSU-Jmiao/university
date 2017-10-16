
# coding: utf-8

# In[32]:


import pickle
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from PIL import Image



# In[2]:


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


# In[3]:


# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
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


# In[4]:


in_shp = list(X_train.shape[1:])
print X_train.shape, in_shp
classes = mods


# In[5]:


def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0 / overSampleFactor)
    sNew = f(tNew)
    return sNew


# In[118]:


def cwt(name,sample):
    iNew = getOverSampledSignal(sample[0], 8)
    qNew = getOverSampledSignal(sample[1], 8)
    width = 32
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


# In[119]:


def preprocess(index, sample):
  name = str(index)+"pic.png"
  cwt(name, sample)


# In[120]:


for index, item in enumerate(X_train):
    preprocess(index,item)
    if index == 50:
      break


# In[ ]:
