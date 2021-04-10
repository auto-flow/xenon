#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-07
# @Contact    : qichun.tang@bupt.edu.cn
from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["FlexibleDecomposer"]


class FlexibleDecomposer(XenonFeatureEngineerAlgorithm):
    class__ = "FlexibleDecomposer"
    module__ = "xenon_ext.decompose"
    # need_y = True
