import math
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from snsdl.keras.wrappers import MlflowClassifier
from myModels.vgg16 import VGG16

# User-defined parameters
imageW = 64
imageH = 64
batch_size = 32

# Image Generators
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

# Space search
paramsSearch = {
    'input_shape':[(imageH, imageW, 3)],
    'num_classes':[train_generator.num_classes],
    'epochs':[2, 5]
}

# Custom model to train
myModel = VGG16(params=paramsSearch)

params = myModel.getParams()

for p in params:

    # Create new classifier
    mlfc = MlflowClassifier(myModel.create_model, **p)

    # Train the model
    history = mlfc.fit_generator(train_generator, val_generator, **p)

    # Predict the test samples
    predict = mlfc.predict_generator(test_generator)

    # Predicted labels
    labels = test_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())
    y_predict = [labels[k] for k in list(np.argmax(predict, axis=-1))]

    # True labels
    y_true = [labels[k] for k in test_generator.classes]

    # Log mlflow
    mlfc.log()
