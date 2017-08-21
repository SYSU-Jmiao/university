
from tensorflow.python.lib.io import file_io
import pickle
import numpy as np
from PIL import Image
from scipy import interpolate, signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io


BUCKET_NAME = "yonidavidson-university"
train_file = "gs://" + BUCKET_NAME + "/data/RML2016.10a_dict.dat"

# Load the dataset ...

f = file_io.FileIO(train_file, mode='r')
Xd = pickle.load(f)
snrs, mods = map(lambda j: sorted(
    list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):
            lbl.append((mod, snr))
X = np.vstack(X)


X_samples = np.expand_dims(X, 3)
print X_samples.shape


def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy) + 1])
    yy1[np.arange(len(yy)), yy] = 1
    return yy1


Y_samples = to_onehot(map(lambda x: mods.index(
    lbl[x][0]), np.arange(X_samples.shape[0])))
print Y_samples.shape


classes = mods
print classes


def getOverSampledSignal(sample, overSampleFactor):
    sampleSize = sample.shape[0]
    t = np.arange(0, sampleSize, 1)
    f = interpolate.InterpolatedUnivariateSpline(t, sample)
    tNew = np.arange(0.0, sampleSize, 1.0 / overSampleFactor)
    sNew = f(tNew)
    return sNew


def cwt(x, buf):
    sNew = getOverSampledSignal(x, 8)
    width = 32
    widths = np.arange(1, width)
    cwtmatr = signal.cwt(sNew, signal.ricker, widths)
    plt.imshow(cwtmatr, extent=[-1, 1, 1, width],
               cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.draw()
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close()
    return Image.open(buf)


def preprocessor(index, x):
    i, q = x[0], x[1]
    vgg16ImageSize = (224, 224)
    finalImage = np.empty((224, 224, 3), dtype=np.uint8)
    with io.BytesIO() as buf1:
        with io.BytesIO() as buf2:
            iImg = cwt(i, buf1)
            qImg = cwt(q, buf2)

            iImg = iImg.resize(vgg16ImageSize)
            grayI = iImg.convert('L')
            qImg = qImg.resize(vgg16ImageSize)
            grayQ = qImg.convert('L')

            h, w = qImg.size
            finalImage[:, :, 0] = grayI
            finalImage[:, :, 1] = grayQ
            if not index % 100:
                print str(index)
            return finalImage


# In[ ]:


X_samples_cwt = map(lambda (index, x): preprocessor(
    index, x), enumerate(X_samples))


# In[ ]:


print X_samples_cwt[0]


# In[ ]:


post_data = {}
post_data["X"] = X_samples_cwt
post_data["Y"] = Y_samples
post_data["classes"] = classes
post_data["snrs"] = snrs


# In[ ]:


pickle.dump(post_data, open("post_data.p", "wb"))


# In[ ]:


with file_io.FileIO("post_data.p", mode='r') as input_f:
    with file_io.FileIO("gs://" + BUCKET_NAME + "/data" + '/post_cwt_data.dat', mode='w+') as output_f:
        output_f.write(input_f.read())
