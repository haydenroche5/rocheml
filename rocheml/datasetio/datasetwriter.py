import os
import h5py


class DatasetWriter:
    def __init__(self, num_rows, cols, hdf_file_path, buffer_size,
                 force=False):
        if not force and os.path.exists(hdf_file_path):
            raise Exception(
                'DatasetWriter::__init__: {} already exists.'.format(
                    hdf_file_path))
        self.hdf_file_path = hdf_file_path
        self.db = h5py.File(self.hdf_file_path, 'w')
        self.datasets = {}
        self.dataset_idxs = {}
        for col in cols:
            dims_with_rows = (num_rows, ) + col['dims']
            dataset = self.db.create_dataset(col['name'],
                                             dims_with_rows,
                                             dtype=col['dtype'])
            self.datasets[col['name']] = dataset
            self.dataset_idxs[col['name']] = 0

        self.buffer = []
        if buffer_size <= 0:
            raise Exception(
                'DatasetWriter::__init__: Buffer size must be > 0.')
        self.buffer_size = buffer_size

    def add(self, row):
        # Don't add the row if the columns don't match the schema.
        if set(row.keys()) != set(self.datasets.keys()):
            raise Exception(
                'DatasetWriter::add: Row columns don\'t match dataset columns. Row columns: {}. Dataset columns {}.'
                .format(row.keys(), self.datasets.keys()))

        self.buffer.append(row)
        # Check if the buffer is full and, if so, flush the buffer.
        if len(self.buffer) == self.buffer_size:
            self.flush()

    def flush(self):
        if self.buffer:
            for col_name, dataset in self.datasets.items():
                num_rows = len(self.buffer)
                curr_idx = self.dataset_idxs[col_name]
                new_idx = self.dataset_idxs[col_name] + num_rows
                for i, row in enumerate(self.buffer):
                    self.datasets[col_name][curr_idx + i] = row[col_name]

                self.dataset_idxs[col_name] = new_idx

        self.buffer = []

    def close(self):
        self.flush()
        self.db.close()
