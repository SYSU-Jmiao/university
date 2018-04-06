'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='transfer_densenet',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Transfer learning using DenseNet on Cloud ML Engine',
      author='Yoni Davidson',
      author_email='yonidavidson@gmail.com',
      license='Unlicense',
      install_requires=[
          'keras',
          'h5py',
          'Pillow',
          'numpy'],
      zip_safe=False)
