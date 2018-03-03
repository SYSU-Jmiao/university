#!/usr/bin/env python 
import sys


def get_classes_from_folders():
    print "hello"
    return []


def create_db(data_dir):
    classes = get_classes_from_folders()
    if len(classes) != 11:
        return 'failed to collect classes'
    return 

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print 'no directorty location exiting...'
        sys.exit(1)
    data_dir = sys.argv[1]
    print "data_dir=" + data_dir
    err = create_db(data_dir)
    if err is None:
        print 'success'
    else:
        print err