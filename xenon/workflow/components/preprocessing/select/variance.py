#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-07
# @Contact    : qichun.tang@bupt.edu.cn
from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm
from .base import SklearnSelectMixin

__all__ = ["VarianceThreshold"]


class VarianceThreshold(XenonFeatureEngineerAlgorithm, SklearnSelectMixin):
    class__ = "VarianceThreshold"
    module__ = "xenon_ext.feature_selection"
    need_y = True
