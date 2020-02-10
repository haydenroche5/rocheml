from tensorflow.keras.utils import to_categorical
import numpy as np
import h5py


# Modified slightly from pyimagesearch's hdf5datasetgenerator.py
class DatasetGenerator:
    def __init__(self,
                 db_file_path,
                 dataset_name,
                 batch_size,
                 feat_key,
                 preprocessors=None,
                 aug=None,
                 binarize=True,
                 classes=2,
                 limit=None,
                 shuffle=False):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.batch_size = batch_size
        self.feat_key = feat_key
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.shuffle = shuffle

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(db_file_path, "r")
        self.dataset = self.db[dataset_name]

        if limit:
            self.num_features = limit
        else:
            self.num_features = self.dataset.shape[0]

        self.idxs = np.arange(0, self.num_features, self.batch_size)
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for idx in self.idxs:
                # extract the features and labels from the HDF dataset
                features = self.dataset[idx:idx + self.batch_size][self.feat_key]
                labels = self.dataset[idx:idx + self.batch_size]['label']

                # check to see if the labels should be binarized
                if self.binarize:
                    labels = to_categorical(labels, self.classes)

            # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # initialize the list of processed features
                    proc_features = []

                    # loop over the features
                    for feature in features:
                        # loop over the preprocessors and apply each
                        # to the feature
                        for p in self.preprocessors:
                            feature = p.preprocess(feature)

                        # update the list of processed features
                        proc_features.append(feature)

                    # update the features array to be the processed
                    # features
                    features = np.array(proc_features)

            # if the data augmenator exists, apply it
                if self.aug is not None:
                    (features, labels) = next(
                        self.aug.flow(features,
                                      labels,
                                      batch_size=self.batch_size))

            # yield a tuple of features and labels
                yield (features, labels)
            # increment the total number of epochs
            epochs += 1
            if self.shuffle:
                np.random.shuffle(self.idxs)

    def close(self):
        # close the database
        self.db.close()