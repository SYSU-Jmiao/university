
# coding: utf-8

# In[1]:


# https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# https://github.com/rajshah4/image_keras/blob/master/notebook.ipynb
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# http://www.codesofinterest.com/2017/08/bottleneck-features-multi-class-classification-keras.html


# In[14]:


get_ipython().system(u'pip install keras')
get_ipython().system(u'pip install h5py')
get_ipython().system(u'pip install missinglink-sdk')


# In[15]:


import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape
from keras.utils.np_utils import to_categorical
import numpy as np
import missinglink


# In[16]:


import h5py
filename = './data_cwt.hdf5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())


# In[17]:


X = f['X_samples']
Y = f['Y_samples']


# In[18]:


np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,int(n_examples)), size=int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))


# In[19]:


def generate_data_from_hdf5(X_list,Y_list ,indexs ,batchSize):
    maxIndex= len(X_list)-batchSize
    print "generator: maxIndex = " + str(maxIndex)
    for i in xrange(0, maxIndex, batchSize):
        current_index = indexs[i:i+batchSize]
        x = map(lambda t: X_list[t], current_index)
        y = map(lambda t: Y_list[t], current_index)
        print "generator: current index = " + str(i)
        yield (np.array(x),np.array(y))


# In[20]:


in_shp = list(X.shape[1:])
print X.shape, in_shp
classes = f['Classes']


# In[21]:


# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


# In[22]:


def save_bottlebeck_features(X, Y, train_idx, test_idx, batch_size):

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator =  generate_data_from_hdf5(X,Y ,train_idx ,batch_size)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = generate_data_from_hdf5(X,Y ,test_idx ,batch_size)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


# In[26]:


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
    print "train"
    print train_data.shape
    print train_labels.shape
    print train_labels[40]

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=11)
    print train_labels.shape
    print train_labels[40]

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    print "validation"
    print validation_data.shape
    print validation_labels.shape
    print validation_labels[40]

    # convert the training labels to categorical vectors
    validation_labels = to_categorical(validation_labels, num_classes=11)
    print validation_labels.shape
    print validation_labels[40]

    top_model = Sequential()
    top_model.add(Flatten(input_shape=train_data.shape[1:], name="flatten1"))
    print top_model.output_shape
    top_model.add(Dense(256, activation='relu', name="dense1"))
    print top_model.output_shape
    top_model.add(Dropout(0.5))
    print top_model.output_shape
    top_model.add(Dense(11, kernel_initializer="he_normal", name="dense2"))
    print top_model.output_shape
    top_model.add(Activation('softmax'))
    print top_model.output_shape
    top_model.add(Reshape([len(classes)], name="reshape2"))
    print top_model.output_shape
    top_model.compile(loss='categorical_crossentropy', optimizer='adam')

    missinglink_callback = missinglink.KerasCallback(owner_id="73b7dbec-273d-c6b7-776d-55812449a4e4", project_token="WxqnIeHhwiLIFejy")
    missinglink_callback.set_properties(display_name='cwt transfer learning', description='basic transfer using cwt preprocessing')
#     filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
#     keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),

    top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=2,
              callbacks = [
#               keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
              keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False),
              missinglink_callback
    ])
    top_model.save_weights(top_model_weights_path)


# In[24]:


save_bottlebeck_features(X, Y,train_idx, test_idx,batch_size )


# In[27]:


train_top_model()


# In[ ]:





# In[ ]: