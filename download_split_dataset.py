from snsdl.utils import SplitDataset

"""
Download a dataset sample and split it in train, test and validation subsets.
"""

def download_input(input_dir):
    import requests
    import os
    url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
    print("downloading '{}' into '{}'".format(url, os.path.join('/tmp', 'flower_photos.tgz')))
    r = requests.get(url)
    with open('flower_photos.tgz', 'wb') as f:
        f.write(r.content)
    import tarfile
    print("decompressing flower_photos.tgz to '{}'".format(output_dir))
    with tarfile.open("flower_photos.tgz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path=input_dir)

input_dir = '/tmp/dataset/input'
output_dir = '/tmp/dataset/output'

download_input(input_dir)

SplitDataset.split(input_dir, output_dir, move=False, balanced=False, test_ratio=0.25, val_ratio=0.2, shuffle=False, verbose=5)