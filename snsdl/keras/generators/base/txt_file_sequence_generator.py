import math
import numpy as np
import os
from . import BaseSequenceGenerator
from keras.utils import Sequence
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences

class TxtFileSequenceGenerator(BaseSequenceGenerator):
    """Generate sequence data from the filesystem for Keras."""

    def __init__(self, X, y, padding_size=0, padding_position='post', text_sep=' ', preprocessors=[], *args, **kwargs):
        """
        Class constructor.

        Attributes:
            X: Array representing the samples data.
            y: Dict mapping samples and labels. {'sample1.txt': 0, 'sample2.txt': 1, 'sample3.txt': 2, 'sample4.txt': 1}
            preprocessors: Array of objects to preprocess the data.
        """

        # Initialize superclass parameters first
        super(TxtFileSequenceGenerator, self).__init__(X, y, *args, **kwargs)

        self.preprocessors = preprocessors
        self.padding_size = padding_size
        self.padding_position = padding_position
        self.text_sep = text_sep
    
    def process(self, id):
        """
        Read and process a csv file.

        Parameters:
            id: csv id.

        Returns:
            Numpy array: image
        """

        with open(id) as f:
            content = " ".join(line.rstrip() for line in f)

        # check to see if our preprocessors are not Empty
        if len(self.preprocessors) > 0:

            # loop over the preprocessors and apply each to the content file
            for p in self.preprocessors:
                content = p.preprocess(content)

        content = content.split(self.text_sep)

        if self.padding_size > 0:
            content = pad_sequences([content], maxlen=self.padding_size, padding='post')[0]
            # TODO - informar com um warning que texto foi truncado se self.padding_size < len(content)

        return np.array(content)