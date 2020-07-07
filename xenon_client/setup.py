#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from setuptools import setup, find_packages

setup(
    name='xenon_cli',
    packages=find_packages(),
    description='Xenon client',
    version='3.0',
    install_requires=['click >= 7.0',
                      'requests >= 2.21.0'],
    entry_points={'console_scripts': ['xenon_cli=xenon_cli:cli', ], })