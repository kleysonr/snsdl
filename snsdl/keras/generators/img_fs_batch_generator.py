import math, os
import numpy as np
from imutils import paths
from snsdl.keras.generators.base import ImgFsSequenceGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

class ImgFsBatchGenerator():
    
    def __init__(self, dataset_path, test_ratio=0.25, val_ratio=0.0, batch_size=32, binary_classification=False, shuffle=False, preprocessors=[]):
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

        self.trainGenerator = None
        self.testGenerator = None
        self.valGenerator = None

        # Dict mapping classes and images path
        self.data = {}

        # Dict for train, test and val dataset
        self.train_test_val = {'train': [], 'test': [], 'val':[]}

        # Dict for train, test and val labels
        self.labels_train_test_val = {'train': [], 'test': [], 'val':[]}

        # Lenght of the dataset
        self.datasetsize = 0

        # Number of images of the smallest class
        self.minsize = math.inf

        # Mapping between class name and labels enconding {class_name:class_code}
        self.encoded_classes = None

        # Mapping between class name its encoded index {class_name:class_index}
        self.class_indices = None

        # Encoder
        self.le = None

        # Get a list of all the images under dataset/{class}/*
        fileslist = paths.list_images(dataset_path)

        for file in fileslist:

            # Extract the label
            label = file.split(os.path.sep)[-2]

            # Populate dict mapping
            try:
                self.data[label]
            except KeyError:
                self.data[label] = []
            finally:
                self.data[label].append(file)
                self.datasetsize += 1

        # Loop over each class
        for k in self.data.keys():

            # Save the size of the smallest class
            self.minsize = len(self.data[k]) if len(self.data[k]) < self.minsize else self.minsize

        # Calculate the offset where test samples begins, based on the smallest class.
        # Force to have balanced classes for training.
        self.offset = int(self.minsize * (1.0 - self.test_ratio))                
    
        # Split the full dataset in train and test datasets
        self.__split_train_test()

        # Split train dataset to generate the validation dataset
        if self.val_ratio > 0:

            X_train, X_val, _, _ = train_test_split(self.train_test_val['train'], self.train_test_val['train'], test_size=val_ratio)

            self.train_test_val['train'] = X_train
            self.train_test_val['val'] = X_val

        # Encode labels
        if binary_classification:
            self.le = LabelEncoder()
        else:
            self.le = LabelBinarizer()

        # Encode classes
        classes_name = list(self.data.keys())
        class_codes = self.le.fit_transform(classes_name)

        # Create mappings
        self.encoded_classes = dict(zip(classes_name, self.le.fit_transform(classes_name)))
        self.class_indices = dict(zip(classes_name, list(np.argmax(class_codes, axis=-1))))        

        # Create dataset labels
        self.__create_labels()

        self.__info()

    def __create_labels(self):

        # Loop over each dataset
        for ds in self.train_test_val.keys():

            # Size of elements in the dataset
            size = len(self.train_test_val[ds])

            labels = []
            for i in range(0, size):

                # Class name
                label = self.train_test_val[ds][i].split(os.path.sep)[-2]
                label = self.encoded_classes[label]
                labels.append(label)

            self.labels_train_test_val[ds] = dict(zip(self.train_test_val[ds], labels))

    def __split_train_test(self):

        _train = []
        _test = []

        # Loop over each class
        for k in self.data.keys():

            # Shuffle the images in each class
            items = self.data[k]
            np.random.shuffle(items)

            _train += items[:self.offset]
            _test  += items[self.offset:]

        np.random.shuffle(_train)
        np.random.shuffle(_test)

        self.train_test_val['train'] = _train
        self.train_test_val['test'] = _test

    @property
    def train(self):
        """Get an instance of a train generator"""

        if self.trainGenerator is None:
            self.trainGenerator = ImgFsSequenceGenerator(self.train_test_val['train'], self.labels_train_test_val['train'], batch_size=self.batch_size, shuffle=self.shuffle, preprocessors=self.preprocessors)

        return self.trainGenerator

    @property
    def test(self):
        """Get an instance of a test generator"""

        if self.testGenerator is None:
            self.testGenerator = ImgFsSequenceGenerator(self.train_test_val['test'], self.labels_train_test_val['test'], batch_size=self.batch_size, shuffle=self.shuffle, preprocessors=self.preprocessors)

        return self.testGenerator

    @property
    def val(self):
        """Get an instance of a validation generator"""

        if self.valGenerator is None:
            self.valGenerator = ImgFsSequenceGenerator(self.train_test_val['val'], self.labels_train_test_val['val'], batch_size=self.batch_size, shuffle=self.shuffle, preprocessors=self.preprocessors)

        return self.valGenerator

    def getNumberOfClasses(self):
        """Get the number of training classes."""
        return len(self.data.keys())

    def getDatasetSize(self, dataset):
        """Get the size of a given dataset (full / train / test / val)."""

        if dataset == 'full':
            return self.datasetsize
        else:
            return len(self.train_test_val[dataset])

    def getTrueClasses(self, dataset):
        """Get an array with the true classes for a given dataset (train / test / val)."""

        encoded = [self.labels_train_test_val[dataset][s] for s in self.train_test_val[dataset]]
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