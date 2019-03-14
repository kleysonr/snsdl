import math
import numpy as np
import os
from . import BaseSequenceGenerator
from keras.utils import Sequence
from keras import backend as K

class ImgFsSequenceGenerator(BaseSequenceGenerator):
    """Generate sequence of images from the filesystem to the Keras."""

    def __init__(self, X, y, preprocessors=[], *args, **kwargs):
        """Class constructor.
        
        Arguments:
            X {list} -- Sample data ids.
            y {list} -- Encoded data labels.
        
        Keyword Arguments:
            preprocessors {list} -- An objects list to preprocess the samples data. (default: {[]})
        """
        
        import cv2

        # Initialize superclass parameters first
        super(ImgFsSequenceGenerator, self).__init__(X, y, *args, **kwargs)

        self.preprocessors = preprocessors
    
    def process(self, id):
        """Read the image and preprocess it.
        
        Arguments:
            id {srt} -- Image id.
        
        Returns:
            [numpy.ndarray] -- Image content.
        """

        # Read image
        image = cv2.imread(id)

        # check to see if our preprocessors are not Empty
        if len(self.preprocessors) > 0:

            # loop over the preprocessors and apply each to the image
            for p in self.preprocessors:
                image = p.preprocess(image)

        return image.astype(K.floatx()) / 255.0