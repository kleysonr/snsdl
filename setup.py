# https://python-packaging.readthedocs.io/en/latest/minimal.html
# python3 setup.py register sdist upload

from setuptools import setup

setup(name='snsdl',
      version='0.0.2',
      description='Deep Learning helpers and mlflow integration',
      url='https://github.com/kleysonr/snsdl',
      author='Kleyson Rios',
      author_email='kleysonr@gmail.com',
      packages=['snsdl'],
      zip_safe=False)
