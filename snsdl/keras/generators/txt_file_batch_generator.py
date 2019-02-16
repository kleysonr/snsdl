import math, os
import numpy as np
from snsdl.utils import paths
from snsdl.utils.splitds import SplitDataset
from snsdl.keras.generators.base.txt_file_sequence_generator import TxtFileSequenceGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import shuffle as skshuffle

class TxtFileBatchGenerator():
    
    def __init__(self, dataset_path, balanced=False, test_ratio=0.25, val_ratio=0.0, batch_size=32, binary_classification=False, shuffle=False, preprocessors=[]):
        """
        Class constructor.

        Attributes:
            dataset_path: Images directory.
            test_ratio: Ratio from the full dataset for testing. If unspecified, test_ratio will default to 0.25.
            val_ratio: Ratio from the train dataset for validation. If unspecified, val_ratio will default to 0.0
            batch_size: Number of samples to be generated. If unspecified, batch_size will default to 32.
            binary_classification: Set True if is binary classification (sigmoid) problem. If binary_classification, test_ratio will default to False.
            shuffle: If true shuffles dataset before each epoch. If unspecified, shuffle will default to False.
            preprocessors: Array of objects to preprocess the data.
        """

        if not (test_ratio > 0.0 and test_ratio < 1.0):
            raise ValueError('test_ratio must be > 0.0 and < 1.0')

        if not (val_ratio >= 0.0 and val_ratio < 1.0):
            raise ValueError('val_ratio must be >= 0.0 and < 1.0')

        if not (batch_size > 0):
            raise ValueError('batch_size must be > 0')

        self.dataset_path = dataset_path
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.binary_classification = binary_classification
        self.preprocessors = preprocessors
        self.shuffle = shuffle
        self.balanced = balanced

        self.trainGenerator = None
        self.testGenerator = None
        self.valGenerator = None

        self._data = SplitDataset.previewSplit(self.dataset_path, balanced=self.balanced, test_ratio=self.test_ratio, val_ratio=self.val_ratio, shuffle=self.shuffle, type='txt')

        # Dict for train, test and val dataset
        self.train_test_val = {'train': [], 'test': [], 'val':[]}

        # Dict for train, test and val labels
        self.labels_train_test_val = {'train': [], 'test': [], 'val':[]}

        # Lenght of the dataset
        self.datasetsize = 0

        # Encoder
        self.le = None
        
        # Encode labels
        if binary_classification:
            self.le = LabelEncoder()
        else:
            self.le = LabelBinarizer()

        # Encode classes
        classes_name = list(self._data.keys())
        class_codes = self.le.fit_transform(classes_name)

        # Create mappings
        self.encoded_classes = dict(zip(classes_name, self.le.fit_transform(classes_name)))
        if binary_classification:
            self.class_indices = dict(zip(classes_name, class_codes))
        else:
            self.class_indices = dict(zip(classes_name, list(np.argmax(class_codes, axis=-1))))

        # Re-organize the list of files
        for label in self._data.keys():
            for ds in self._data[label]:
                for f in self._data[label][ds]:
                    self.train_test_val[ds].append(f)
                    self.labels_train_test_val[ds].append(self.encoded_classes[label])
                    self.datasetsize += 1

        # Shuffle the arrays
        self.train_test_val['train'], self.labels_train_test_val['train'] = skshuffle(self.train_test_val['train'], self.labels_train_test_val['train'], random_state=0)
        self.train_test_val['test'], self.labels_train_test_val['test'] = skshuffle(self.train_test_val['test'], self.labels_train_test_val['test'], random_state=0)
        self.train_test_val['val'], self.labels_train_test_val['val'] = skshuffle(self.train_test_val['val'], self.labels_train_test_val['val'], random_state=0)

        self.__info()

    @property
    def train(self):
        """Get an instance of a train generator"""

        if self.trainGenerator is None:
            self.trainGenerator = TxtFileSequenceGenerator(self.train_test_val['train'], self.labels_train_test_val['train'], batch_size=self.batch_size, shuffle=self.shuffle, preprocessors=self.preprocessors)

        return self.trainGenerator

    @property
    def test(self):
        """Get an instance of a test generator"""

        if self.testGenerator is None:
            self.testGenerator = TxtFileSequenceGenerator(self.train_test_val['test'], self.labels_train_test_val['test'], batch_size=self.batch_size, shuffle=self.shuffle, preprocessors=self.preprocessors)

        return self.testGenerator

    @property
    def val(self):
        """Get an instance of a validation generator"""

        if self.valGenerator is None:
            self.valGenerator = TxtFileSequenceGenerator(self.train_test_val['val'], self.labels_train_test_val['val'], batch_size=self.batch_size, shuffle=self.shuffle, preprocessors=self.preprocessors)

        return self.valGenerator

    def getNumberOfClasses(self):
        """Get the number of training classes."""
        return len(self._data.keys())

    def getDatasetSize(self, dataset):
        """Get the size of a given dataset (full / train / test / val)."""

        if dataset == 'full':
            return self.datasetsize
        else:
            return len(self.train_test_val[dataset])

    def getTrueClasses(self, dataset):
        """Get an array with the true classes for a given dataset (train / test / val)."""

        encoded = [self.labels_train_test_val[dataset][s] for i, s in enumerate(self.train_test_val[dataset])]
        encoded_indx = list(np.argmax(encoded, axis=-1))

        return [list(self.class_indices.keys())[list(self.class_indices.values()).index(s)] for s in encoded_indx]

    def __info(self):
        print('')
        print('Dataset information:')
        print('   Dataset size: {}'.format(self.getDatasetSize('full')))
        print('   Training size: {}'.format(self.getDatasetSize('train')))
        print('   Testing size: {}'.format(self.getDatasetSize('test')))
        print('   Validation size: {}'.format(self.getDatasetSize('val')))
        print('   # of classes: {}'.format(self.getNumberOfClasses()))
        print('')