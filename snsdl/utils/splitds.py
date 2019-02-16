import os
import shutil
import sys
import numpy as np
from snsdl.utils import paths

class SplitDataset():

    @staticmethod
    def split(inputdir, outputdir, move=False, balanced=False, test_ratio=0.25, val_ratio=0.0, shuffle=False, verbose=-1):
        """
        Split a dataset stored in the filesystem in training, testing and validation datasets.

        Parameters:
            inputdir: path for the dataset.
            outputdir: path for the split dataset.
            move: Set True to move the file instead of copying them. If unspecified, move will default to False. 
            balanced: Set True to create balanced training classes. If unspecified, balanced will default to False.
            test_ratio: Ratio from the full dataset for testing. If unspecified, test_ratio will default to 0.25.
            val_ratio: Ratio from the train dataset for validation. If unspecified, val_ratio will default to 0.0.
            shuffle: If true shuffles dataset before the splitting. If unspecified, shuffle will default to False.
            verbose: Prints the execution progress. Set values > 0.

        Returns:
            None
        """

        # Parameters validations
        if not (test_ratio > 0.0 and test_ratio < 1.0):
            raise ValueError('test_ratio must be > 0.0 and < 1.0')

        if not (val_ratio >= 0.0 and val_ratio < 1.0):
            raise ValueError('val_ratio must be >= 0.0 and < 1.0')

        datasize, data = SplitDataset.__readFilesDir(inputdir, shuffle)

        # Create balanced/imbalanced training classes.
        if balanced:
            data = SplitDataset.__balancedSplit(datasize, data, test_ratio, val_ratio)
        else:
            data = SplitDataset.__imbalancedSplit(datasize, data, test_ratio, val_ratio)

        SplitDataset.__splitData(datasize, data, outputdir, move, verbose)

        print("\n[INFO] Finished!")

    @staticmethod
    def previewSplit(inputdir, balanced=False, test_ratio=0.25, val_ratio=0.0, shuffle=False, type='img'):

        # Parameters validations
        if not (test_ratio > 0.0 and test_ratio < 1.0):
            raise ValueError('test_ratio must be > 0.0 and < 1.0')

        if not (val_ratio >= 0.0 and val_ratio < 1.0):
            raise ValueError('val_ratio must be >= 0.0 and < 1.0')

        datasize, data = SplitDataset.__readFilesDir(inputdir, shuffle, type=type)

        # Create balanced/imbalanced training classes.
        if balanced:
            data = SplitDataset.__balancedSplit(datasize, data, test_ratio, val_ratio)
        else:
            data = SplitDataset.__imbalancedSplit(datasize, data, test_ratio, val_ratio)

        return data        

    @staticmethod
    def __splitData(datasize, data, outputdir, move, verbose):

        totalimages = sum(datasize.values())

        try:
            os.makedirs(outputdir, exist_ok=False)
        except FileExistsError as e:
            raise Exception("Make sure that the outputdir doens't exists") from e

        i = 1
        for c in data.keys():
            for ds in data[c]:

                dstpath = os.path.join(outputdir,ds,c)
                os.makedirs(dstpath, exist_ok=False)

                for f in data[c][ds]:

                    _, filename = os.path.split(f)
                    dstfile = os.path.join(dstpath,filename)

                    if move:
                        shutil.move(f, dstfile)
                    else:
                        shutil.copy(f, dstfile)

                    # show an update every `verbose` images
                    if verbose > 0 and i > 0 and (i) % verbose == 0:
                        print("[INFO] processed {}/{}".format(i, totalimages))

                    i += 1

    @staticmethod
    def __balancedSplit(datasize, data, test_ratio, val_ratio):
        """
        Get a balanced subdivision of the data in the following structure:
        {
            class1: {
                train: [img1.jpg, img2.jpg, ...]
                test : [img3.jpg, img4.jpg, ...]
                val  : [...]
            }
            class2: {
                train: [img5.jpg, img6.jpg, ...]
                test : [img7.jpg, img8.jpg, ...]
                val  : [...]
            }
            classN: {
                train: [img9.jpg , img10.jpg, ...]
                test : [img11.jpg, img12.jpg, ...]
                val  : [...]
            }
        }
        """

        _data = {}

        # Number of elements for the smallest class
        minsize = sorted(datasize.items(), key=lambda kv: kv[1])[0][1]

        # Calculate the offset where test samples begins, based on the smallest class.
        offset = int(minsize * (1.0 - test_ratio))

        # Loop over each class
        for c in datasize.keys():

            try:
                _data[c]
            except KeyError:
                _data[c] = {}
                _data[c]['train'] = []
                _data[c]['test'] = []
                _data[c]['val'] = []
            finally:
                _data[c]['train'] = data[c][:offset]
                _data[c]['test'] = data[c][offset:]

            if val_ratio > 0.0:

                val_size = round(offset * val_ratio)

                _data[c]['val'] = _data[c]['train'][:val_size]
                _data[c]['train'] = _data[c]['train'][val_size:]

        return _data

    @staticmethod
    def __imbalancedSplit(datasize, data, test_ratio, val_ratio):
        """
        Get an imbalanced subdivision of the data in the following structure:
        {
            class1: {
                train: [img1.jpg, img2.jpg, ...]
                test : [img3.jpg, img4.jpg, ...]
                val  : [...]
            }
            class2: {
                train: [img5.jpg, img6.jpg, ...]
                test : [img7.jpg, img8.jpg, ...]
                val  : [...]
            }
            classN: {
                train: [img9.jpg , img10.jpg, ...]
                test : [img11.jpg, img12.jpg, ...]
                val  : [...]
            }
        }
        """

        _data = {}

        # Loop over each class
        for c in datasize.keys():

            class_size = datasize[c]

            test_size = round(class_size * test_ratio)
            train_size = int(class_size - test_size)

            try:
                _data[c]
            except KeyError:
                _data[c] = {}
                _data[c]['train'] = []
                _data[c]['test'] = []
                _data[c]['val'] = []
            finally:
                _data[c]['train'] = data[c][:train_size]
                _data[c]['test'] = data[c][train_size:]

            if val_ratio > 0.0:

                val_size = round(train_size * val_ratio)

                _data[c]['val'] = _data[c]['train'][:val_size]
                _data[c]['train'] = _data[c]['train'][val_size:]

        return _data

    @staticmethod
    def __readFilesDir(inputdir, shuffle, type='img'):
        """
        Read the filenames inside a directory organized by folders representing each classes.
        The directory must follow the structure:
           /base_folder
             /class1
               - image1.jpg
               - image2.jpg
             /class2
               - image3.jpg
               - image4.jpg
             ...

        Parameters:
            inputdir: path for the dataset.

        Returns:
            Dict: dictionary mapping class name and number of samples for the class.
            Dict: dictionary mapping class name and fullpath for all images belonging the class.
        """

        data = {}
        size = {}

        # Get a list of all the images under dataset/{class}/*
        fileslist = None
        if type == 'txt':
            fileslist = paths.list_txts(inputdir)
        else:
            fileslist = paths.list_images(inputdir)

        for file in fileslist:

            # Extract the label
            label = file.split(os.path.sep)[-2]

            # Populate dict mapping
            try:
                data[label]
                size[label]
            except KeyError:
                data[label] = []
                size[label] = 0
            finally:
                data[label].append(file)
                size[label] += 1

        if len(size.keys()) == 0:
            print("\n[INFO] Exiting... no input files found!\n")
            sys.exit(0)

        if shuffle:

            for d in data.keys():
                np.random.shuffle(data[d])

        return size, data