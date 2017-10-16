
# coding: utf-8

# In[1]:


# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# https://github.com/rajshah4/image_keras/blob/master/notebook.ipynb
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html


# In[2]:


get_ipython().system(u'pip install keras')
get_ipython().system(u'pip install h5py')
get_ipython().system(u'pip install missinglink-sdk')


# In[4]:


import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape
from keras.utils.np_utils import to_categorical
import numpy as np
import missinglink


# In[5]:


import h5py
filename = '../data_cwt.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())


# In[6]:


X = f['X_samples']
Y = f['Y_samples']


# In[7]:


np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,int(n_examples)), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))


# In[8]:


def generate_data_from_hdf5(X_list,Y_list ,indexs ,batchSize):
    maxIndex= len(X_list)-batchSize
    print "generator: maxIndex = " + str(maxIndex)
    for i in xrange(0, maxIndex, batchSize):
        current_index = indexs[i:i+batchSize]
        x = map(lambda t: X_list[t], current_index)
        y = map(lambda t: Y_list[t], current_index)
        print "generator: current index = " + str(i)
        yield (np.array(x),np.array(y))


# In[9]:


in_shp = list(X.shape[1:])
print X.shape, in_shp
classes = f['Classes']


# In[10]:


# dimensions of our images.
img_width, img_height = 224, 224

model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
train_samples = 100000
validation_samples = 100000
epochs = 50
batch_size = 100


# In[11]:


model_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[ ]:


top_model = Sequential()
top_model.add(Flatten(input_shape=model_vgg.output_shape[1:], name="flatten1"))
top_model.add(Dense(256, activation='relu', name="dense1"))
top_model.add(Dropout(0.5))
top_model.add(Dense(11, kernel_initializer="he_normal", name="dense2"))
top_model.add(Activation('softmax'))
top_model.add(Reshape([len(classes)], name="reshape2"))
# top_model.load_weights('../transfer-cwt/bottleneck_fc_model.h5') // try remove pre-training


model = Model(inputs = model_vgg.input, outputs = top_model(model_vgg.output))
for layer in model.layers[:15]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam')


# In[ ]:


missinglink_callback = missinglink.KerasCallback(owner_id="73b7dbec-273d-c6b7-776d-55812449a4e4", project_token="WxqnIeHhwiLIFejy")
missinglink_callback.set_properties(display_name='cwt transfer learning tuned', description='fine tuning transfer using cwt preprocessing')

train_generator = generate_data_from_hdf5(X, Y, train_idx, batch_size)
validation_generator = generate_data_from_hdf5(X,Y, test_idx, batch_size)

model.fit_generator(
          train_generator,
          steps_per_epoch=train_samples // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_samples// batch_size,
          verbose=2,
          callbacks = [
          keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False),
          missinglink_callback
])
model.save_weights(model_weights_path)


# In[ ]:





# In[ ]:
