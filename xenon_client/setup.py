#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from setuptools import setup, find_packages

setup(
    name='xenon_cli',
    version='0.1.0',
    packages=find_packages(),
    author='qichun tang',
    author_email='tqichun@gmail.com',
    description='Xenon client',
    python_requires='>=3.6.*',
    install_requires=['click >= 7.0',
                      'requests >= 2.21.0'],
    entry_points={'console_scripts': ['xenon_cli=xenon_cli:cli', ], },
    platforms=['Linux'],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)