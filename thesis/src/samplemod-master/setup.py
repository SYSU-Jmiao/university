# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='simpleWaveFinder',
    version='0.0.1',
    description='Find a source of wave',
    long_description=readme,
    author='Yoni Davidson',
    author_email='yonidavidson@gmail.com',
    url='tbd',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

