#!/usr/bin/env python 
import sys
import glob
from os import path


def get_classes_from_folders(data_dir):
    classes_dir = path.join(data_dir,'data/*')
    classes =  glob.glob(classes_dir)
    print "classes:" + str(classes) + ", location:" + classes_dir
    return classes

def get_data_sets(data_location):
    return [],[]

def copy_data(class_name, output_dir, train_data_set, validation_data_set):
    return

def create_db(data_dir, output_dir):
    classes = get_classes_from_folders(data_dir)
    if len(classes) != 11:
        return 'failed to collect classes'
    for c in classes:
        train_data_set, validation_data_set = get_data_sets(c)
        copy_data(c, output_dir, train_data_set, validation_data_set)
    return 

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'no directorty/output location exiting...'
        sys.exit(1)
    data_dir = sys.argv[1]
    output_dir = sys.argv[2]
    print "data_dir=" + data_dir
    err = create_db(data_dir, output_dir)
    if err is None:
        print 'success'
    else:
        print err