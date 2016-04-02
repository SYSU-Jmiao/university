import sys
import os
sys.path.append("../")
import my_utils as mu
import unittest
import numpy as np

#globals
current_dir = os.path.dirname(__file__)


class TDDReadDataFromFiles(unittest.TestCase):
    def test_get_full_file_data(self):
        file_name = current_dir + '/QAM16_signalG_500Ms.csv'
        print(file_name)
        file_size = 250000
        calc = mu.readfile(file_name)
        result = len(calc)
        self.assertEqual(file_size, result)

    def test_get_resampled_file_data(self):
        sampling_size = 500
        file_name = current_dir + '/QAM16_signalG_500Ms.csv'
        calc = mu.readfile(file_name, True, sampling_size)
        result = len(calc)
        self.assertEqual(sampling_size, result)

    def test_matrix_from_files(self):
        sampling_size = 5
        folder_name = current_dir + '/original'
        files_list = mu.listfilesinfolder(folder_name)
        mt = mu.create_matrix_from_filelist(files_list, sampling_size)
        np.set_printoptions(threshold=np.inf)
        print(mt.shape)
        print(mt)
        # result = mt.values.shape()
        # print(result)
