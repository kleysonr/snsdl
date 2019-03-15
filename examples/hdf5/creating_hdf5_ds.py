"""
This example will use the HDF5Writer class to create a new HDF5 database file
holding 3 classes.
"""

import h5py
from snsdl.io.hdf5 import HDF5Writer

# Define the 3 classes with their data.
# Each data sample is composed by 3 columns (3-dim).
class1 = [
    [101, 'Class 1 sample 01', '01'],
    [102, 'Class 1 sample 02', '02'],
    [103, 'Class 1 sample 03', '03'],
    [104, 'Class 1 sample 04', '04'],
    [105, 'Class 1 sample 05', '05'],
    [106, 'Class 1 sample 06', '06'],
    [107, 'Class 1 sample 07', '07'],
    [108, 'Class 1 sample 08', '08'],
    [109, 'Class 1 sample 09', '09'],
    [110, 'Class 1 sample 10', '10'],
]

class2 = [
    [201, 'Class 2 sample 01', '01'],
    [202, 'Class 2 sample 02', '02'],
    [203, 'Class 2 sample 03', '03'],
    [204, 'Class 2 sample 04', '04'],
    [205, 'Class 2 sample 05', '05'],
    [206, 'Class 2 sample 06', '06'],
    [207, 'Class 2 sample 07', '07'],
    [208, 'Class 2 sample 08', '08'],
    [209, 'Class 2 sample 09', '09'],
    [210, 'Class 2 sample 10', '10'],
    [211, 'Class 2 sample 11', '11'],
    [212, 'Class 2 sample 12', '12'],
    [213, 'Class 2 sample 13', '13'],
    [214, 'Class 2 sample 14', '14'],
]

class3 = [
    [301, 'Class 3 sample 01', '01'],
    [302, 'Class 3 sample 02', '02'],
    [303, 'Class 3 sample 03', '03'],
    [304, 'Class 3 sample 04', '04'],
    [305, 'Class 3 sample 05', '05'],
    [306, 'Class 3 sample 06', '06'],
    [307, 'Class 3 sample 07', '07'],
    [308, 'Class 3 sample 08', '08'],
    [309, 'Class 3 sample 09', '09'],
    [310, 'Class 3 sample 10', '10'],
    [311, 'Class 3 sample 11', '11'],
    [312, 'Class 3 sample 12', '12'],
    [313, 'Class 3 sample 13', '13'],
    [314, 'Class 3 sample 14', '14'],
    [315, 'Class 3 sample 15', '15'],
    [316, 'Class 3 sample 16', '16'],
    [317, 'Class 3 sample 17', '17'],
    [318, 'Class 3 sample 18', '18'],
]

# Create a new HDF5Writer object.
# Each sample is a 3-dimensional size.
hdf5db = HDF5Writer('/tmp/rawsample.hdf5', 3)

# Write the data using the method add.
hdf5db.add('class1', class1)
hdf5db.add('class2', class2)
hdf5db.add('class3', class3)

# Flush to disk any buffer un-empty and close the file.
hdf5db.finish()

# Open the HDF5 file and check its content.
f = h5py.File('/tmp/rawsample.hdf5', 'r')

for ds in f.keys():
    print('Dataset {} size: [{},{}]'.format(ds, f[ds].shape[0], f[ds].shape[1]))

f.close()
