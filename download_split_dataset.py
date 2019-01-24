from snsdl.utils import SplitDataset

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
        tar.extractall(path=input_dir)

input_dir = '/tmp/dataset/input'
output_dir = '/tmp/dataset/output'

download_input(input_dir)

SplitDataset.split(input_dir, output_dir, move=False, balanced=False, test_ratio=0.25, val_ratio=0.2, shuffle=False, verbose=5)