#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import shutil

from xenon import ResourceManager
import os

def get_mock_resource_manager():
    mock_path="/tmp/mock_xenon"
    if os.path.exists(mock_path):
        shutil.rmtree(mock_path)
    return ResourceManager(store_path=mock_path)
