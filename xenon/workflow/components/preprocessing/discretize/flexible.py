#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-07
# @Contact    : qichun.tang@bupt.edu.cn
from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["FlexibleDiscretizer"]


class FlexibleDiscretizer(XenonFeatureEngineerAlgorithm):
    class__ = "FlexibleDiscretizer"
    module__ = "xenon_ext.discretize"
    need_y = True
