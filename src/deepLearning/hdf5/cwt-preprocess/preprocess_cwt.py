
# coding: utf-8

# In[13]:


import pickle
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime



# In[14]:


import os
import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue


def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper


# In[15]:


# Load the dataset ...
Xd = pickle.load(open("./RML2016.10a_dict.dat",'rb'))
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
n_examples = X.shape[0]


X_snr = map(lambda x: lbl[x][1], range(0,n_examples))
X_mod = map(lambda x: lbl[x][0], range(0,n_examples))


# In[16]:


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y = to_onehot(map(lambda x: mods.index(lbl[x][0]), range(0,n_examples)))


# In[17]:


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


# In[18]:


def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0 / overSampleFactor)
    sNew = f(tNew)
    return sNew


# In[19]:


@processify
def cwt(name,sample):
#     print ("PROCESS:" + str(os.getpid()))
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

    plt.close(f)


# In[20]:


def getPath(index, snrs, mods):
  path = "./data/" + str(mods[index]) + "/" + str(index)+ "_" + str(mods[index]) + "_" + str(snrs[index]) + ".png"
  return path


# In[21]:


import os
def mkdir(path):
  if not os.path.exists(path):
    os.makedirs(path)


# In[22]:


mkdir("./data")
map(lambda x: mkdir("./data/"+x), mods)


# In[23]:


for index, item in enumerate(X):
    if index <= 216247:
        continue
    path = getPath(index, X_snr, X_mod)
    cwt(path, item)
    if index%1000 == 0:
      print (str(index) + ":" + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:





# In[ ]:
