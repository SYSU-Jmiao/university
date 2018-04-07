'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='group-classification',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='classify in inner group',
      author='Yoni Davidson',
      author_email='yonidavidson@gmail.com',
      license='Unlicense',
      install_requires=[
          'keras',
          'comet_ml',
          'h5py',
          'matplotlib'
      ],
      zip_safe=False)
