import os
import h5py
import numpy as np
from .base_writer import HDF5BaseWriter
import logging

logger = logging.getLogger(__name__)

class HDF5Writer(HDF5BaseWriter):

    def __init__(self, dbPath, featureVectorSize, featureVectorDType='str', estNumSamples=500, maxBufferSize=50000, dbResizeFactor=2):
        """Class constructor.
        
        Arguments:
            dbPath {str} -- Path to where the features database will be stored.
            featureVectorSize {int} -- Number of features for each sample (# of columns).
        
        Keyword Arguments:
            featureVectorDtype {str} -- Feature data type. (default: {str})
            estNumSamples {int} -- Approximate number of samples in the dataset. (default: {500})
            maxBufferSize {int} -- Maximum number of feature vectors to be stored in memory until the buffer is flushed to HDF5. (default: {50000})
            dbResizeFactor {int} -- Factor to resize the database when needed to expand it. (default: {2})
        """

        logger.info('Creating a new HDF5 writer for `{}`.'.format(dbPath))

		# Call the parent constructor
        super(HDF5Writer, self).__init__(dbPath, estNumSamples=estNumSamples, maxBufferSize=maxBufferSize, dbResizeFactor=dbResizeFactor)

        self.featureVectorSize = featureVectorSize
        self.featureVectorDType = featureVectorDType

        # check to see if the output path exists, and if so, raise an exception
        if os.path.exists(dbPath):
            raise ValueError("The supplied `dbPath` already "
                            "exists and cannot be overwritten. Manually delete "
                            "the file before continuing.", dbPath)

        # Open the HDF5 database for writing and initialize the datasets within the group
        self.db = h5py.File(self.dbPath, mode="w")

        # Initialize the total number of features in the buffer
        self.totalFeatures = 0     

        # Initialize the buffer
        self.datasetBuffer = {}

        # HDF5 dataset object handlers
        self.datasets = {}       

    def add(self, datasetName, rows):
        """Add a new entry in a given dataset.
        
        Arguments:
            datasetName {str} -- Name of the dataset inside the hdf5 db.
            rows {array[array]} -- Data representing the sample's feature.
        """

        rows = np.vstack(rows)

		# Add the sample into the buffer
        try:
            self.datasetBuffer[datasetName].extend(rows)
        except:
            # Initialize the buffer
            self.datasetBuffer[datasetName] = []
            self.datasetBuffer[datasetName].extend(rows)

            # Create the dataset
            self._createDatasets(datasetName)

            # Initiliaze dataset index count
            self.idxs[datasetName] = 0

        # Update the number of samples in the buffer
        self.totalFeatures += len(rows)

		# Check to see if we have reached the maximum buffer size
        if self.totalFeatures >= self.maxBufferSize:

			# write the buffers to file
            self._writeBuffers()

    def _createDatasets(self, datasetName):
        """Create a new dataset inside the hdf5 db.
        
        Arguments:
            datasetName {str} -- Name of the dataset inside the hdf5 db.
        """

        # Set the feature vector dtype
        if self.featureVectorDType == 'str':
            dt = h5py.special_dtype(vlen=str)
        else:
            dt = self.featureVectorDType

		# Initialize the datasets
        logger.debug('Creating dataset `{}`.'.format(datasetName))

        if self.featureVectorSize > 1:
            self.datasets[datasetName] = self.db.create_dataset(str(datasetName), (self.estNumSamples, self.featureVectorSize), maxshape=(None, self.featureVectorSize), dtype=dt)
        else:
            self.datasets[datasetName] = self.db.create_dataset(str(datasetName), (self.estNumSamples,), maxshape=(None,), dtype=dt)

    def _writeBuffers(self):
        """Trigger the process of flushing the buffers."""

        logger.info('Writing buffers to disk...')

        for ds in self.datasets.keys():

            if len(self.datasetBuffer[ds]) > 0:

                # write the buffers to disk
                self._writeBuffer(self.datasets[ds], ds, self.datasetBuffer[ds])

                # increment the indexes
                self.idxs[ds] += len(self.datasetBuffer[ds])

                # Reset the buffers and feature counts
                self.datasetBuffer[ds] = []

        self.totalFeatures = 0

    def finish(self):
        """Write un-empty buffers and close the hdf5 db."""

		# Write any unempty buffers to file
        logger.info('Writing un-empty buffers...')
        self._writeBuffers()

		# Compact datasets
        logger.info('Compacting datasets...')
        for ds in self.datasets.keys():

            self._resizeDataset(self.datasets[ds], ds, finished=self.idxs[ds])

		# close the database
        self.db.close()

    def shuffle(self, datasetNames=[], exclude_prefix=None):
        """[summary]
        
        Keyword Arguments:
            datasetNames {list} -- Database keys. (default: {[]})
            exclude_prefix {str} -- Prefix to exclude shuffle from databases keys starting with. (default: {'_'})
        """

        logger.info('Starting database shuffle.')

		# Write any unempty buffers to file
        logger.info('Writing un-empty buffers...')
        self._writeBuffers()        

        if len(datasetNames) == 0:
            datasetNames = self.db.keys()

        for ds in datasetNames:
            if exclude_prefix is not None and not ds.startswith(exclude_prefix):
                try:
                    self.db[ds]
                    logger.info('Shuffling database key `{}`.'.format(ds))
                    np.random.shuffle(self.db[ds])
                except:
                    logger.error('Dataset key `{}` does not exist.'.format(ds))

    def storeArray(self, data, dsName='_data'):
        """[summary]
        
        Arguments:
            data {[type]} -- [description]
        
        Keyword Arguments:
            dsName {str} -- [description] (default: {'_data'})
        """

        logger.debug('Storing `{}` into database.'.format(dsName))

        if type(data) is list:
            data = np.array(data)

        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset(dsName, (data.shape), dtype=dt)
        labelSet[:] = data