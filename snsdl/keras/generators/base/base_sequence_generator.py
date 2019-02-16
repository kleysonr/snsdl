import math
import numpy as np
from keras.utils import Sequence

class BaseSequenceGenerator(Sequence):
    """Base sequence generator for Keras."""
    
    def __init__(self, X, y, batch_size=32, shuffle=False):
        """
        Class constructor.

        Attributes:
            X: Array representing the sample data ids.
            y: Dict mapping samples and labels. {'sample1': 0, 'sample2': 1, 'sample3': 2, 'sample4': 1}
            batch_size: Number of samples to be generated. If unspecified, batch_size will default to 32.
            shuffle: If true shuffles dataset before each epoch. If unspecified, shuffle will default to False.
        """

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Get the number of batchs per epoch."""

        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data"""

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_ids = [self.X[k] for k in indexes]
        y_ids = [self.y[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_ids, y_ids, index)

        return (X, y)

    def __data_generation(self, batch_ids, y_ids, index):
        """
        Generates data containing batch_size samples.

        Parameters:
            batch_ids: Array representing the batch sample ids.
            index: position of the batch in the Sequence.

        Returns:
            Numpy array: sample data
            Numpy array: labels
        """

        _X = []
        _y = []

        # Get labels
        for i, ID in enumerate(batch_ids):
            _X.append(batch_ids)
            _y.append(y_ids[i])

        return self.__process(np.array(batch_ids), np.array(_y), index)

    def on_epoch_end(self):
        """Updates indexes and shuffle the data after each epoch"""

        self.indexes = np.arange(len(self.X))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __process(self, X, y, index):
        """
        Get the raw data and preprocess them.

        Parameters:
            X: Numpy array representing the batch sample ids.
            y: Numpy array representing the data labels.
            index: position of the batch in the Sequence.

        Returns:
            Numpy array: sample data
            Numpy array: labels
        """

        images = []
        labels = []

        for i, file in enumerate(X):

            image = self.process(str(file))

            images.append(image)
            labels.append(y[i])

            # print('Batch: {}-{} >> {}'.format(index, i, str(file)))
            # print('Batch: '.format(str(file)))

        return np.array(images), np.array(labels)          

    def process(self, id):
        """
        Read and process the image id.

        Parameters:
            id: image id.

        Returns:
            Numpy array: image
        """
        raise RuntimeError('Must be implemented by subclasses.')