from comet_ml import Experiment

import argparse
from datetime import datetime
from os import path, mkdir
from subprocess import check_call

import keras
import numpy as np
from keras import models
from keras.layers import Dropout, Flatten, Dense, Reshape, ZeroPadding2D, Conv2D, Activation
import cPickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import uuid


def get_data(data_location, local_data):
    if path.exists(local_data):
        print("data exists, continue.: " + data_location)
    else:
        print("getting data from: " + data_location)
        check_call(['gsutil', '-m', '-q', 'cp', '-r', data_location, local_data])
        check_call(['ls', '/tmp/'])


def train(local_data, job_dir):
    # Load the dataset ...
    #  You will need to seperately download or generate this file
    Xd = cPickle.load(open(local_data, 'rb'))
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)

    # Partition the data
    #  into training and test sets of the form we can train/test on
    #  while keeping SNR and Mod labels handy for each
    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = n_examples * 0.5
    train_idx = np.random.choice(range(0, int(n_examples)), size=int(n_train), replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy) + 1])
        yy1[np.arange(len(yy)), yy] = 1
        return yy1

    Y_train = to_onehot(map(lambda x: mods.index(lbl[x][0]), train_idx))
    Y_test = to_onehot(map(lambda x: mods.index(lbl[x][0]), test_idx))

    in_shp = list(X_train.shape[1:])
    print(X_train.shape, in_shp)
    classes = mods

    # Build VT-CNN2 Neural Net model using Keras primitives --
    #  - Reshape [N,2,128] to [N,1,2,128] on input
    #  - Pass through 2 2DConv/ReLu layers
    #  - Pass through 2 Dense layers (ReLu and Softmax)
    #  - Perform categorical cross entropy optimization

    dr = 0.5  # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape(in_shp + [1], input_shape=in_shp))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(256, (1, 3), kernel_initializer="glorot_uniform", name="conv1", activation="relu", padding="valid"))
    model.add(Dropout(dr))
    model.add(ZeroPadding2D((0, 2)))
    model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer="he_normal", activation="relu", name="dense1"))
    model.add(Dropout(dr))
    model.add(Dense(11, kernel_initializer="he_normal", name="dense2"))
    model.add(Activation('softmax'))
    model.add(Reshape([len(classes)]))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()

    # Set up some params
    nb_epoch = 100  # number of epochs to train on
    batch_size = 200  # training batch size

    # perform training ...
    #   - call the main training loop in keras for our network+dataset
    filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
    history = model.fit(X_train,
                        Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        callbacks=[
                            keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto'),
                            keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
                        ])
    # we re-load the best weights once training is finished
    model.load_weights(filepath)

    # Show simple version of performance
    score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
    print(score)


    # ****************#
    meta_graphs = "/tmp/meta"
    mkdir(meta_graphs)

    plt.figure()
    plt.title('Training performance')
    plt.plot(history.epoch, history.history['loss'], label='train loss+error')
    plt.plot(history.epoch, history.history['val_loss'], label='val_error')
    plt.legend()

    plt.savefig(path.join(meta_graphs,"training_performance.png"))

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(path.join(meta_graphs, str(uuid.uuid4())+"_confusion_matrix.png"))

    # Plot confusion matrix
    test_Y_hat = model.predict(X_test, batch_size=batch_size)
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, X_test.shape[0]):
        j = list(Y_test[i, :]).index(1)
        k = int(np.argmax(test_Y_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plot_confusion_matrix(confnorm, labels=classes)

    # Plot confusion matrix
    acc = {}
    for snr in snrs:
        # extract classes @ SNR
        test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = Y_test[np.where(np.array(test_SNRs) == snr)]
        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])
        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.argmax(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plt.figure()
        plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print( "Overall Accuracy: ", cor / (cor + ncor))
        acc[snr] = 1.0 * cor / (cor + ncor)

    print(acc)
    fd = open(path.join(meta_graphs,'results_cnn2_d0.5.dat'), 'wb')
    cPickle.dump(("CNN2", 0.5, acc), fd)

    # Plot accuracy curve
    plt.plot(snrs, map(lambda x: acc[x], snrs))
    plt.xlabel("Signal to Noise Ratio")
    plt.ylabel("Classification Accuracy")
    plt.title("CNN2 Classification Accuracy on RadioML 2016.10 Alpha")
    plt.savefig(path.join(meta_graphs, "Accuracy.png"))

    check_call(['ls', meta_graphs])
    check_call(['gsutil', 'cp', '-r', meta_graphs, job_dir])


# Create a function to allow for different training data and other options
def train_model(data_location='data/',
                job_dir='./job_dir', **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    local_data = path.join("/tmp", "data_dict.dat")
    get_data(data_location, local_data)
    train(local_data, job_dir)


if __name__ == '__main__':
    # Add the following code anywhere in your machine learning file
    experiment = Experiment(api_key="xlfxZoR6K87Hd3t1xTKiI6N44")

    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-location',
        help='Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
