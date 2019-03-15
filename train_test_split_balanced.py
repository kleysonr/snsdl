"""
This example will read a HDF5 with datasets/class and split them
into train, test [and val] datasets in a new HDF5 file.

The training classes will be balanced based on the smaller class.
"""

import h5py
from snsdl.io.hdf5.utils import TrainingDataset

tds = TrainingDataset('/tmp/rawsample.hdf5', '/tmp/dl_ds.hdf5', 3, balanced=True, test_ratio=0.25, val_ratio=0.15)
tds.generate(batchSizePerClass=5, shuffle=False)

# Checking
f = h5py.File('/tmp/dl_ds.hdf5', 'r')

print('\nTraining dataset:')
print(f['train'][:])

print('\nTesting dataset:')
print(f['test'][:])

print('\nValidation dataset:')
try:
    print(f['val'][:])
except:
    pass

f.close()