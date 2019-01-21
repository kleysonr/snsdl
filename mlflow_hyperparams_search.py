import math
import os
import tempfile
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
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

# Space search
paramsSearch = {
    'input_shape':[(imageH, imageW, 3)],
    'num_classes':[train_generator.num_classes],
    'epochs':[2, 5]
}

# Custom model to train
myModel = VGG16(params=paramsSearch)

# Get all the combinations of the parameters
params = myModel.getSearchParams()

for p in params:

    artifacts_dir = tempfile.mkdtemp()

    # Create new classifier
    mlfc = MlflowClassifier(myModel.create_model, **p)

    # Train the model
    history = mlfc.fit_generator(train_generator, val_generator, **p)

    # Predict the test samples
    predict = mlfc.predict_generator(test_generator)

    # Score
    score = mlfc.evaluate_generator(test_generator)

    # Training / Validation / Testing metrics
    metrics = mlfc.getMetricsValues(history, score=score)

    # Predicted labels
    labels = test_generator.class_indices
    labels = dict((v,k) for k,v in labels.items())
    y_predict = [labels[k] for k in list(np.argmax(predict, axis=-1))]

    # True labels
    y_true = [labels[k] for k in test_generator.classes]

    # Evaluate results
    class_names = sorted(test_generator.class_indices.items(), key=lambda kv: kv[1])
    class_names = [item[0] for item in class_names]

    Eval.plot_history(history, png_output=os.path.join(artifacts_dir,'images'), show=False)
    Eval.full_multiclass_report(y_true, y_predict, class_names, png_output=os.path.join(artifacts_dir,'images'), show=False)

    # Log mlflow
    mlfc.log(artifacts=artifacts_dir, metrics=metrics)
