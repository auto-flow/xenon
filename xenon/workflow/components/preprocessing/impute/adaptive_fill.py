#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["AdaptiveSimpleImputer"]


class AdaptiveSimpleImputer(XenonFeatureEngineerAlgorithm):
    class__ = "AdaptiveSimpleImputer"
    module__ = "xenon.feature_engineer.impute"
