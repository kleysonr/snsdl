import math
import numpy as np
import os
import cv2
from . import BaseSequenceGenerator
from keras.utils import Sequence
from keras import backend as K

class ImgFsSequenceGenerator(BaseSequenceGenerator):
    """Generate sequence data from the filesystem for Keras."""

    def __init__(self, X, y, preprocessors=[], *args, **kwargs):
        """
        Class constructor.

        Attributes:
            X: Array representing the samples data.
            y: Dict mapping samples and labels. {'sample1.jpg': 0, 'sample2.jpg': 1, 'sample3.jpg': 2, 'sample4.jpg': 1}
            preprocessors: Array of objects to preprocess the data.
        """

        # Initialize superclass parameters first
        super(ImgFsSequenceGenerator, self).__init__(X, y, *args, **kwargs)

        self.preprocessors = preprocessors
    
    def process(self, id):
        """
        Read and process the image id.

        Parameters:
            id: image id.

        Returns:
            Numpy array: image
        """

        # Read image
        image = cv2.imread(id)

        # check to see if our preprocessors are not Empty
        if len(self.preprocessors) > 0:

            # loop over the preprocessors and apply each to the image
            for p in self.preprocessors:
                image = p.preprocess(image)

        return image.astype(K.floatx()) / 255.0