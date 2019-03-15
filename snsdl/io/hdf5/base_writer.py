import os
import h5py
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HDF5BaseWriter:

    def __init__(self, dbPath, estNumSamples=500, maxBufferSize=50000, dbResizeFactor=2):
        """Class constructor.
        
        Arguments:
            dbPath {str} -- Path to where the features database will be stored.
        
        Keyword Arguments:
            estNumSamples {int} -- Approximate number of samples in the dataset. (default: {500})
            maxBufferSize {int} -- Maximum number of feature vectors to be stored in memory until the buffer is flushed to HDF5. (default: {50000})
            dbResizeFactor {int} -- Factor to resize the database when needed to expand it. (default: {2})
        """

        self.dbPath = dbPath
        self.estNumSamples = estNumSamples
        self.maxBufferSize = maxBufferSize
        self.dbResizeFactor = dbResizeFactor

        # Indexes dictionary to control datasets append
        self.idxs = {}

    def _writeBuffers(self):
        raise RuntimeError('Must be implemented by subclasses.')

    def _writeBuffer(self, dataset, datasetName, buf, sparse=False):
        """Write the data in the buffers to disk.
        
        Arguments:
            dataset {[type]} -- [description]
            datasetName {[type]} -- [description]
            buf {[type]} -- [description]
        
        Keyword Arguments:
            sparse {bool} -- [description] (default: {False})
        """

		# If the buffer is a list, then compute the ending index based on
		# the lists length
        if type(buf) is list:
            end = self.idxs[datasetName] + len(buf)

		# Otherwise, assume that the buffer is a NumPy/SciPy array, so
		# compute the ending index based on the array shape
        else:
            end = self.idxs[datasetName] + buf.shape[0]

		# Check to see if the dataset needs to be resized
        if end > dataset.shape[0]:
            logger.debug('Triggering `{}` db resize'.format(datasetName))
            self._resizeDataset(dataset, datasetName, baseSize=end)

		# if this is a sparse matrix, then convert the sparse matrix to a
		# dense one so it can be written to file
        if sparse:
            buf = buf.toarray()

		# dump the buffer to file
        logger.debug('Writing `{}` buffer'.format(datasetName))
        dataset[self.idxs[datasetName]:end] = buf

    def _resizeDataset(self, dataset, dbName, baseSize=0, finished=0):
        """[summary]
        
        Arguments:
            dataset {[type]} -- [description]
            dbName {[type]} -- [description]
        
        Keyword Arguments:
            baseSize {int} -- [description] (default: {0})
            finished {int} -- [description] (default: {0})
        """

		# grab the original size of the dataset
        origSize = dataset.shape[0]

		# check to see if we are finished writing rows to the dataset, and if
		# so, make the new size the current index
        if finished > 0:
            newSize = finished

		# otherwise, we are enlarging the dataset so calculate the new size
		# of the dataset
        else:
            newSize = baseSize * self.dbResizeFactor

		# determine the shape of (to be) the resized dataset
        shape = list(dataset.shape)
        shape[0] = newSize

		# show the old versus new size of the dataset
        dataset.resize(tuple(shape))
        logger.debug('old size of `{}`: {:,}; new size: {:,}'.format(dbName, origSize, newSize))
