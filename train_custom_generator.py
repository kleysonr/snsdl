import keras
import cv2
import math
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from snsdl.keras.generators import ImgFsBatchGenerator
from snsdl.evaluation import Eval

"""
For a big dataset that doesn't fit in the memory, you need to train the model
using a generator.

This code is an example of using a custom image generator to feed the training
processes with batches of images.

You might also preprocess the images if desired.

At the end, some reports are saved in the output directory.
"""

# Configuration
imageW = 64
imageH = 64
batch_size = 32
epochs = 1

# Preprocessor for image resizing
class ImageResizePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        # store the target image width, height, and interpolation
        # method used when resizing
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        # resize the image to a fixed size, ignoring the aspect
        # ratio
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

# ImageResizePreprocessor instance
irp = ImageResizePreprocessor(imageW, imageH)

# Keras batch generator instance
batchGen = ImgFsBatchGenerator('/tmp/dataset/input/flower_photos', test_ratio=0.20, val_ratio=0.2, preprocessors=[irp], batch_size=batch_size)

# Dataset generators
train_generator = batchGen.train
test_generator = batchGen.test
val_generator = batchGen.val

# Model parameters
input_shape = (imageH, imageW, 3)
num_classes = batchGen.getNumberOfClasses()

# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Train model on dataset
history = model.fit_generator(
            generator=train_generator,
            validation_data=test_generator,
            use_multiprocessing=False,
            workers=1)

# Save model
os.makedirs('saved_models', exist_ok=True)
model.save('saved_models/cnn_example.model')

# Predict new samples
steps = math.ceil(batchGen.getDatasetSize('val') / batch_size)
predict = model.predict_generator(val_generator, steps=steps)

# Predicted labels
labels = (batchGen.class_indices)
labels = dict((v,k) for k,v in labels.items())
y_predict = [labels[k] for k in list(np.argmax(predict, axis=-1))]

# GT labels
y_true = batchGen.getTrueClasses('val')

# Evaluate results
Eval.plot_history(history, png_output='output/', show=False)
Eval.full_multiclass_report(y_true, y_predict, batchGen.le.classes_, png_output='output/', show=False)
