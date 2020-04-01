import unittest
from datasetio.datasetwriter import DatasetWriter
from datasetio.datasetgenerator import DatasetGenerator
import h5py
import os
import numpy as np
import string
import random
import math


class TestDatasetGenerator(unittest.TestCase):
    def setUp(self):
        self.feat_length = 5
        self.seq_length = 5
        self.buffer_size = 5
        self.num_rows = 50
        self.dataset_file_path = 'test.hdf'
        self.dataset_name = 'test'

        self.dtypes = [('feat_seq', 'float', (self.seq_length,
                                              self.feat_length)),
                       ('label', 'int'), ('file', h5py.string_dtype())]
        self.dataset_writer = DatasetWriter('test', self.num_rows, self.dtypes,
                                            self.dataset_file_path,
                                            self.buffer_size)
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

    def check_db(self, batch_size, expected_rows, shuffle):
        gen = DatasetGenerator(self.dataset_file_path,
                               self.dataset_name,
                               batch_size,
                               'feat_seq',
                               shuffle=shuffle)
        gen_features = []
        gen_labels = []
        for features, labels in gen.generator(1):
            gen_features.extend(features.tolist())
            gen_labels.extend(labels.tolist())
        self.assertEqual(len(expected_rows), len(gen_labels))

        for gen_label, gen_features in zip(gen_labels, gen_features):
            result = [
                row for row in expected_rows if row['label'] == gen_label
                and np.array_equal(row['feat_seq'], gen_features)
            ]
            self.assertTrue(result)

    def test_full(self):
        expected_rows = self.initialize_expected_rows()
        for i in range(0, self.num_rows):
            row = self.generate_random_row()
            expected_rows[i] = row
            self.dataset_writer.add(row)
        self.dataset_writer.close()

        batch_size = 3
        self.check_db(batch_size, expected_rows, False)

    def test_full_shuffle(self):
        expected_rows = self.initialize_expected_rows()
        for i in range(0, self.num_rows):
            row = self.generate_random_row()
            expected_rows[i] = row
            self.dataset_writer.add(row)
        self.dataset_writer.close()

        batch_size = 3
        self.check_db(batch_size, expected_rows, True)


if __name__ == '__main__':
    unittest.main()
