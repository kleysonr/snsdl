import math
import os
import tempfile
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from snsdl.evaluation import Eval
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

# Set callback functions to early stop training and save the best model so far
# callbacks = [[EarlyStopping(monitor='val_loss', patience=5), 
#                 ModelCheckpoint(filepath='/tmp/best_model.h5', monitor='val_loss', save_best_only=True)]]

# Space search
paramsSearch = {
    'input_shape':[(imageH, imageW, 3)],
    'num_classes':[train_generator.num_classes],
    'epochs':[2, 5]
    # ,'callbacks':callbacks
}

# Custom model to train
myModel = VGG16(params=paramsSearch)

# Get all the combinations of the parameters
params = myModel.getSearchParams()

for p in params:

    artifacts_dir = tempfile.mkdtemp()

    # Create new classifier
    mlfc = MlflowClassifier(myModel.create_model, train_generator, test_generator, val_generator, **p)

    # Train the model
    history = mlfc.fit_generator()

    # Predict the test/val samples
    mlfc.predict_generator()

    # Get the training, validation and testing metrics
    metrics = mlfc.getMetricsValues()

    # Predicted labels for test set
    y_predict = mlfc.getTestPredictLabels()

    # True labels of the test set
    y_true = mlfc.getTestTrueLabels()

    # Evaluate results
    class_names = mlfc.getClassNames()

    Eval.plot_history(history, png_output=os.path.join(artifacts_dir,'images'), show=False)
    Eval.full_multiclass_report(y_true, y_predict, class_names, png_output=os.path.join(artifacts_dir,'images'), show=False)

    # Classification Report
    Eval.classification_report(y_true, y_predict, output_dir=artifacts_dir)

    # Log mlflow
    mlfc.log(artifacts=artifacts_dir, metrics=metrics)
