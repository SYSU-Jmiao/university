import sys
sys.path.append("../")
import my_utils as mu
import unittest


class TDDReadDataFromFiles(unittest.TestCase):
    def test_get_full_file_data(self):
        file_name = 'QAM16_signalG_500Ms.csv'
        file_size = 250000
        calc = mu.readfile(file_name)
        result = len(calc)
        print("number of sampels equeal:" + str(result))
        self.assertEqual(file_size, result)

    def test_get_resampled_file_data(self):
        sampling_size = 500
        calc = mu.readfile('QAM16_signalG_500Ms.csv', True, sampling_size)
        result = len(calc)
        print("number of sampels equeal:" + str(result))
        self.assertEqual(sampling_size, result)
