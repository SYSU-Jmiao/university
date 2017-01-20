import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


class __DataLocation:
    def __init__(self):
        self.original = "/dev/null"
        self.filtered = "/dev/null"
        self.originalfiles = []
        self.filteredfiles = []


def getlocationofdata():
    """ parses data location from command line arguments. """

    parser = argparse.ArgumentParser(description='Compare tested data to\
             syntatic waves.')
    parser.add_argument('origin', nargs=1, help='\
                        location of the source sample files')
    parser.add_argument('filtered', nargs=1, help='\
                        location of the filtered sample files')
    args = parser.parse_args()
    d = __DataLocation
    d.origin = args.origin[0]
    d.filtered = args.filtered[0]
    d.originfiles = listfilesinfolder(d.origin)
    d.filteredfiles = listfilesinfolder(d.filtered)
    return d


def listfilesinfolder(folder_name):
    """ Return a list of files in folder. """

    return glob.glob(folder_name+"/*")


def readlistoffiles(list_of_files, resample=False, resamplesize=0):
    """ Reads an array of files list and puts them in a map. """

    new_dict = dict()
    for f in list_of_files:
        new_dict[f] = readfile(f, resample, resamplesize)
    return new_dict


def create_matrix_from_filelist(files_list, sampling_size):
    """ Return matrix ( (sampling_size), len(files_list)) of vectores. """

    m = np.zeros((len(files_list), (sampling_size)))
    for idx, val in enumerate(files_list):
        v = readfile(val, True, sampling_size)[:, 1]
        m[idx] = v[0]
    return m.T


def readfile(dest_file, resample=False, resamplesize=0):
    data = np.genfromtxt(dest_file, delimiter=',')
    if resample:
        data = signal.resample(data, resamplesize)
    return data


def plot(data_vector):
    """ Simple plotting of data. """

    plt.plot(data_vector)
    plt.ylabel('some numbers')
    plt.show()
