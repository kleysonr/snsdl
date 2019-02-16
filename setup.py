# https://python-packaging.readthedocs.io/en/latest/minimal.html
# python3 setup.py register sdist upload

from setuptools import setup, find_packages

setup(name='snsdl',
      version='0.0.3',
      description='Deep Learning helpers and mlflow integration',
      url='https://github.com/kleysonr/snsdl',
      author='Kleyson Rios',
      author_email='kleysonr@gmail.com',
      packages=find_packages(),
      zip_safe=False)
