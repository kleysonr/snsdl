import math
import numpy as np
from keras.utils import Sequence

class BaseSequenceGenerator(Sequence):
    """Base sequence generator for Keras."""
    
    def __init__(self, X, y, batch_size=32, shuffle=False, **kwargs):
        """Class constructor.
        
        Arguments:
            X {list} -- Sample data ids.
            y {list} -- Encoded data labels.
        
        Keyword Arguments:
            batch_size {int} -- Number of samples to be generated. (default: {32})
            shuffle {bool} -- Shuffles the dataset before each epoch. (default: {False})
        """

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.class_indices = kwargs['class_indices']
        self.classes = kwargs['classes']
        self.filenames = kwargs['filenames']
        self.on_epoch_end()

    def __len__(self):
        """Get the number of batchs per epoch.
        
        Returns:
            int -- Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch. It should typically be equal to ceil(num_samples / batch_size).
        """

        return math.ceil(len(self.X) / self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data.
        
        Arguments:
            index {int} -- Batch index. Position of the batch in the Sequence.
        
        Returns:
            (numpy.ndarray, numpy.ndarray) -- Tuple of preprocessed images and their labels for a given batch id.
        """

        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batch_ids = [self.X[k] for k in indexes]
        y_ids = [self.y[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_ids, y_ids, index)

        return (X, y)

    def __data_generation(self, batch_ids, y_ids, index):
        """Generates data containing batch_size samples.
        
        Arguments:
            batch_ids {list} -- Samples ids for a given batch id.
            y_ids {list} -- Encoded labels for a given batch id.
            index {int} -- Batch index. Position of the batch in the Sequence.
        
        Returns:
            (numpy.ndarray, numpy.ndarray) -- Tuple of preprocessed images and their labels for a given batch id.
        """

        _X = []
        _y = []

        # Get labels
        for i, ID in enumerate(batch_ids):
            _X.append(batch_ids)
            _y.append(y_ids[i])

        return self.__process(np.array(batch_ids), np.array(_y), index)

    def on_epoch_end(self):
        """Updates indexes and shuffle the data after each epoch."""

        self.indexes = np.arange(len(self.X))

        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __process(self, X, y, index):
        """Get the raw data and preprocess them.
        
        Arguments:
            X {numpy.ndarray} -- Samples ids for a given batch id.
            y {numpy.ndarray} -- Encoded labels for a given batch id.
            index {int} -- Batch index. Position of the batch in the Sequence.
        
        Returns:
            (numpy.ndarray, numpy.ndarray) -- Tuple of preprocessed images and their labels for a given batch id.
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
        """This method must be implemented by a subclasse to read a given
        sample based on an id and preprocessed it if needed.

        Arguments:
            id {srt} -- Sample id.
        
        Returns:
            [numpy.ndarray] -- Sample content.
        """

        raise RuntimeError('Must be implemented by subclasses.')