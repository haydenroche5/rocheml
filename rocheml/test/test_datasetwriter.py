import unittest
from datasetio.datasetwriter import DatasetWriter
import h5py
import os
import numpy as np
import string
import random


class TestDatasetWriter(unittest.TestCase):
    def setUp(self):
        self.feat_length = 10
        self.seq_length = 20
        self.buffer_size = 5
        self.num_rows = 100
        self.dataset_file_path = 'test.hdf'

        self.dtypes=[('feat_seq', 'float', (self.seq_length, self.feat_length)), ('label', 'int'), ('file', h5py.string_dtype())]
        self.dataset_writer = DatasetWriter('test', self.num_rows, self.dtypes, self.dataset_file_path, self.buffer_size)
        self.taken_files = set()

    def tearDown(self):
        os.remove(self.dataset_file_path)

    def initialize_expected_rows(self):
        expected_rows = []
        for i in range(0, self.num_rows):
            zero_features = np.zeros((self.seq_length, self.feat_length))
            row = self.generate_row(zero_features, 0, '')
            expected_rows.append(row)

        return expected_rows

    def generate_row(self, features, label, file):
        return {'feat_seq': features, 'label': label, 'file': file}

    def generate_random_row(self):
        features = np.random.rand(self.seq_length, self.feat_length)
        label = np.random.randint(2)
        letters = string.ascii_lowercase

        # Generate a unique file name, i.e. one that hasn't been used in this test yet.
        file = ''.join(random.choice(letters) for i in range(10)) + '.mp4'
        while file in self.taken_files:
            file = ''.join(random.choice(letters) for i in range(10)) + '.mp4'
        self.taken_files.add(file)

        return {'feat_seq': features, 'label': label, 'file': file}

    def check_equality(self, expected_row, actual_row):
        expected_row_tuple = tuple([expected_row[name] for name in [dtype[0] for dtype in self.dtypes]])
        actual_row_tuple = tuple(actual_row)
        
        for expected_val, actual_val in zip(expected_row_tuple, actual_row_tuple):
            if isinstance(expected_val, np.ndarray):
                if not np.array_equal(expected_val, actual_val):
                    return False
            else:
                if expected_val != actual_val:
                    return False

        return True

    def check_db(self, expected_rows):
        db = h5py.File(self.dataset_file_path, 'r')
        actual_rows = db['test']
        for expected_row, actual_row in zip(expected_rows, actual_rows):
            self.assertTrue(self.check_equality(expected_row, actual_row))

    def test_empty(self):
        expected_rows = self.initialize_expected_rows()
        self.check_db(expected_rows)

    def test_add_one_less_than_buffer_size(self):
        expected_rows = self.initialize_expected_rows()
        for i in range(0, self.buffer_size - 1):
            row = self.generate_random_row()
            expected_rows[i] = row
            self.dataset_writer.add(row)
        self.dataset_writer.close()

        self.check_db(expected_rows)

    def test_add_one_more_than_buffer_size(self):
        expected_rows = self.initialize_expected_rows()
        for i in range(0, self.buffer_size + 1):
            row = self.generate_random_row()
            expected_rows[i] = row
            self.dataset_writer.add(row)
        self.dataset_writer.close()

        self.check_db(expected_rows)

    def test_full(self):
        expected_rows = self.initialize_expected_rows()
        for i in range(0, self.num_rows):
            row = self.generate_random_row()
            expected_rows[i] = row
            self.dataset_writer.add(row)
        self.dataset_writer.close()

        self.check_db(expected_rows)


if __name__ == '__main__':
    unittest.main()
