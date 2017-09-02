'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='radioML_cwt',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Radio ML using wavelet preprocess',
      author='Yoni Davidson',
      author_email='yonidavidson@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'matplotlib',
          'missinglink-sdk',
          'Pillow',
          'h5py'],
      zip_safe=False)
