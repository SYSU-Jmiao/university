"""Find matching wave by simple correlation."""

import argparse
import glob
import numpy as np
import os


class __DataLocation:
    def __init__(self):
        self.original = "/dev/null"
        self.filtered = "/dev/null"
        self.originalfiles = []
        self.filteredfiles = []


def __getlocationofdata():
    parser = argparse.ArgumentParser(description='Compare tested data to\
             syntatic waves.')
    parser.add_argument('origin', nargs=1, help='lo\
                        cation of the source sample files')
    parser.add_argument('filtered', nargs=1, help='lo\
                        cation of the filtered sample files')
    args = parser.parse_args()
    d = __DataLocation
    d.origin = args.origin[0]
    d.filtered = args.filtered[0]
    d.originfiles = glob.glob(d.origin + "/*")
    d.filteredfiles = glob.glob(d.filtered + "/*")
    return d


def __readlistoffiles(list_of_files):
    new_dict = dict()
    for f in list_of_files:
        new_dict[f] = __readfile(f)
        print("key:" + f + ", number of elements:" + str(len(new_dict[f])))
    return new_dict


def __readfile(dest_file):
    print("loading:" + dest_file)
    data = np.genfromtxt(dest_file, delimiter=',')
    return data


def __print_best_match(name, candidate, possibilities):
    number_of_samples = 20000
    max_value = 0
    max_name = "none"
    sig_noise = np.resize(candidate, number_of_samples)
    for k in possibilities:
        sig = np.resize(possibilities[k], number_of_samples)
        corr = np.corrcoef(sig, sig_noise)
        score = corr[0, 1]
        if score > max_value:
            max_name = k
            max_value = score
    print(os.path.basename(name) + "==" + os.path.basename(max_name))


def main():
    """Script starts here."""
    locations = __getlocationofdata()
    print(locations.__dict__)
    originscsvs = __readlistoffiles(locations.originfiles)
    filterdcsvs = __readlistoffiles(locations.filteredfiles)
    for k in originscsvs:
        __print_best_match(k, originscsvs[k], filterdcsvs)

if __name__ == "__main__":
    main()
