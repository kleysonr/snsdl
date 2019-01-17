import keras
import cv2
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from snsdl.evaluation import Eval

# Configuration
imageW = 64
imageH = 64
batch_size = 32
epochs = 2

idg = ImageDataGenerator(rescale=1. / 255)

train_generator = idg.flow_from_directory(
    directory="/tmp/dataset/train",
    target_size=(imageH, imageW),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = idg.flow_from_directory(
    directory="/tmp/dataset/test",
    target_size=(imageH, imageW),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = idg.flow_from_directory(
    directory="/tmp/dataset/val",
    target_size=(imageH, imageW),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

# Model parameters
input_shape = (imageH, imageW, 3)
num_classes = train_generator.num_classes

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
            validation_data=val_generator,
            use_multiprocessing=False,
            workers=1)

# Save model
model.save('model/cnn_example.model')

# Predict new samples
steps = math.ceil(test_generator.n / batch_size)
predict = model.predict_generator(test_generator, steps=steps)

# Predicted labels
labels = test_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
y_predict = [labels[k] for k in list(np.argmax(predict, axis=-1))]

# GT labels
y_true = [labels[k] for k in test_generator.classes]

# Evaluate results
class_names = sorted(test_generator.class_indices.items(), key=lambda kv: kv[1])
class_names = [item[0] for item in class_names]
Eval.plot_history(history, png_output='output/', show=False)
Eval.full_multiclass_report(y_true, y_predict, class_names, png_output='output/', show=False)
