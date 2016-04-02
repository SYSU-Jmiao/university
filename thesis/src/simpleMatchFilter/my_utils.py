import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt


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
    d.originfiles = glob.glob(d.origin + "/*")
    d.filteredfiles = glob.glob(d.filtered + "/*")
    return d


def readlistoffiles(list_of_files, resample=False, resamplesize=0):
    """ Reads an array of files list and puts them in a map. """

    new_dict = dict()
    for f in list_of_files:
        new_dict[f] = readfile(f, resample, resamplesize)
        print("key:" + f + ", number of elements:" + str(len(new_dict[f])))
    return new_dict


def readfile(dest_file, resample=False, resamplesize=0):
    print("loading:" + dest_file)
    data = np.genfromtxt(dest_file, delimiter=',')
    if resample:
        print("resample requested:" + str(resamplesize))
    return data


def plot(data_vector):
    """ Simple plotting of data. """

    plt.plot(data_vector)
    plt.ylabel('some numbers')
    plt.show()
