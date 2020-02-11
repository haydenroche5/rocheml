import os
import h5py
import numpy as np

class DatasetWriter:
    def __init__(self, dataset_name, num_rows, dtypes, hdf_file_path, buffer_size,
                 force=False):
        if not force and os.path.exists(hdf_file_path):
            raise Exception(
                'DatasetWriter::__init__: {} already exists.'.format(
                    hdf_file_path))
        self.hdf_file_path = hdf_file_path
        self.db = h5py.File(hdf_file_path, 'w')
        self.dtypes = dtypes
        self.dataset = self.db.create_dataset(dataset_name, (num_rows,), dtype=dtypes)

        if buffer_size <= 0:
            raise Exception(
                'DatasetWriter::__init__: Buffer size must be > 0.')
        self.buffer_size = buffer_size
        self.buffer = np.recarray((buffer_size,), dtype=dtypes)
        self.buffer_idx = 0
        self.dataset_idx = 0

    def add(self, row):
        if self.dataset_idx == len(self.dataset):
            print(f'Dataset full. Not adding request row: {row}.')
            return

        # Check if the buffer is full and, if so, flush the buffer.
        if self.buffer_idx == self.buffer.shape[0]:
            self.flush()
        row_tuple = tuple([row[name] for name in list(self.buffer.dtype.names)])
        self.buffer[self.buffer_idx] = row_tuple
        self.buffer_idx += 1


    def clear_buffer(self):
        self.buffer_idx = 0
        self.buffer = np.recarray((self.buffer_size,), dtype=self.dtypes)

    def flush(self):
        if self.buffer_idx != 0:
            self.dataset[self.dataset_idx:self.dataset_idx+self.buffer_idx] = self.buffer[0:self.buffer_idx]
            self.dataset_idx += self.buffer_idx
            self.clear_buffer()

    def close(self):
        self.flush()
        self.db.close()
