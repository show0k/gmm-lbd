# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.rst").read()
except IOError:
    long_description = ""

setup(
    name="gmm-lbd",
    version="0.0.3",
    description="Some gmm regression experiments for learning by demonstration",
    url="https://github.com/show0k/gmm-lbd",
    license="MIT",
    author="Théo Segonds",
    author_email='theo.segonds@inria.fr',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'sklearn', 'pypot'],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
    ]
)
