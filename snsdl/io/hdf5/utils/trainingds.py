import h5py
import os
import numpy as np
from snsdl.io.hdf5 import HDF5Writer
import logging

logger = logging.getLogger(__name__)

class TrainingDataset():

    def __init__(self, inputHDF5, outputHDF5, featuresVectorSize, balanced=False, test_ratio=0.25, val_ratio=0.0):
        """Class constructor.
        
        Arguments:
            inputHDF5 {str} -- Full path for a HDF5 file. Each class MUST be a separated dataset inside the database. Dataset's started with `_` will be ignored.
            outputHDF5 {} -- [description]
            featuresVectorSize {[type]} -- [description]
        
        Keyword Arguments:
            balanced {bool} -- [description] (default: {False})
            test_ratio {float} -- [description] (default: {0.25})
            val_ratio {float} -- [description] (default: {0.0})
        
        Raises:
            ValueError -- [description]
            an -- [description]
            ValueError -- [description]
        """

        self.inputHDF5 = inputHDF5
        self.outputHDF5 = outputHDF5
        self.featuresVectorSize = 1 + featuresVectorSize # Add an additional column (first) reserved for the label.
        self.balanced = balanced
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

        # check to see if the input path not exists
        if not os.path.exists(inputHDF5):
            raise ValueError("The supplied `inputHDF5` does not exists.", inputHDF5)

        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(outputHDF5):
            raise ValueError("The supplied `outputHDF5` already "
                            "exists and cannot be overwritten. Manually delete "
                            "the file before continuing.", outputHDF5)

    def generate(self, batchSizePerClass=1000, shuffle=True):

        self.inputdb = h5py.File(self.inputHDF5, 'r')
        self.outputdb = HDF5Writer(self.outputHDF5, self.featuresVectorSize, estNumSamples=50000, maxBufferSize=10000)

        # Get classes and their sizes
        self.class_size = self.__get_class_size()

        # Create balanced/imbalanced training classes.
        if self.balanced:
            self.__balancedDS(batchSizePerClass, shuffle)
        else:
            self.__imbalancedDS(batchSizePerClass, shuffle)

        mapping = np.array(list(self.class_size.items()))
        self.outputdb.storeArray(mapping, dsName='_classes')

        self.inputdb.close()
        self.outputdb.finish()

    def __balancedDS(self, batchSize, shuffle):

        # Get the number of elements for the smallest class
        minsize = sorted(self.class_size.items(), key=lambda kv: kv[1])[0][1]

        # Calculate the offset where test samples begins, based on the smallest class.
        train_offset = int(minsize * (1.0 - self.test_ratio))

        # Calculate the offset where validation samples begins.
        val_offset = round(train_offset * self.val_ratio) if self.val_ratio > 0.0 else 0

        logger.info('Generating balanced training dataset...')

        for i in range(0, train_offset-val_offset, batchSize):

            _buffer = []

            # Loop over all the classes to split data
            for ds in self.inputdb.keys():
                start = i
                end = min(train_offset-val_offset, i+batchSize)

                data = self.inputdb[ds][start:end]
                _buffer.extend(np.concatenate([np.array([ds] * len(data)).reshape(len(data),1), data], axis=1))

            _buffer = np.vstack(_buffer)

            if shuffle:
                np.random.shuffle(_buffer)

            self.outputdb.add('train', _buffer)

        logger.info('Generating balanced testing dataset...')

        for i in range(train_offset, self.class_size[ds], batchSize):

            _buffer = []

            # Loop over all the classes to split data
            for ds in self.inputdb.keys():
                start = i
                end = min(self.class_size[ds], i+batchSize)

                data = self.inputdb[ds][start:end]
                _buffer.extend(np.concatenate([np.array([ds] * len(data)).reshape(len(data),1), data], axis=1))

            _buffer = np.vstack(_buffer)

            if shuffle:
                np.random.shuffle(_buffer)

            self.outputdb.add('test', _buffer)

        if self.val_ratio > 0.0:

            logger.info('Generating balanced validation dataset...')

            val_startoffset = round(train_offset * (1.0 - self.val_ratio)) if self.val_ratio > 0.0 else 0

            for i in range(val_startoffset, train_offset, batchSize):

                _buffer = []

                # Loop over all the classes to split data
                for ds in self.inputdb.keys():
                    start = i
                    end = min(train_offset, i+batchSize)

                    data = self.inputdb[ds][start:end]
                    _buffer.extend(np.concatenate([np.array([ds] * len(data)).reshape(len(data),1), data], axis=1))

                _buffer = np.vstack(_buffer)

                if shuffle:
                    np.random.shuffle(_buffer)

                self.outputdb.add('val', _buffer)

    def __imbalancedDS(self, batchSize, shuffle):
        pass

    def __get_class_size(self):

        classes = {}

        # Loop over all the classes
        for ds in self.inputdb.keys():
            classes[ds] = self.inputdb[ds].shape[0]

        return classes