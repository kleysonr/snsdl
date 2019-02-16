from snsdl.utils import SplitDataset

"""
Split a given dataset in train, test and validation subsets.
"""

SplitDataset.split('/project/dataset/input', '/project/dataset/output', move=False, balanced=False, test_ratio=0.25, val_ratio=0.0, shuffle=False, verbose=5)