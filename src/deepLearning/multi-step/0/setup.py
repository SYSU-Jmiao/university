'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='base-model',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Base model - classification of 11 classes',
      author='Yoni Davidson',
      author_email='yonidavidson@gmail.com',
      license='Unlicense',
      install_requires=[
          'keras',
          ],
      zip_safe=False)
