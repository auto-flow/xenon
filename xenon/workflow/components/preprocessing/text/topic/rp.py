#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com


from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["RpTransformer"]


class RpTransformer(XenonFeatureEngineerAlgorithm):
    module__ = "xenon.feature_engineer.text.topic"
    class__ = "RpTransformer"
