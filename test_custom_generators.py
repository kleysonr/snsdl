from snsdl.keras.generators import TxtFileBatchGenerator, ImgFsBatchGenerator
import numpy as np

"""
Use this to test the custom generator.
"""


a = TxtFileBatchGenerator('/data/tmp/dataset_txt', balanced=True, test_ratio=0.4, val_ratio=0.5, batch_size=3, binary_classification=True)
# a = ImgFsBatchGenerator('/data/tmp/dataset_img', balanced=True, test_ratio=0.4, val_ratio=0.5, batch_size=3)

data = []     # store all the generated data batches
labels = []   # store all the generated label batches
for d, l in a.test:
    data.append(d)
    labels.append(l)

data = np.array(data)
data = np.reshape(data, (data.shape[0]*data.shape[1],) + data.shape[2:])

labels = np.array(labels)
labels = np.reshape(labels, (labels.shape[0]*labels.shape[1],) + labels.shape[2:])    
    
print(data.shape, labels.shape)