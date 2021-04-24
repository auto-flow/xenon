#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-24
# @Contact    : qichun.tang@bupt.edu.cn

from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["FMClassifier"]


class FMClassifier(XenonClassificationAlgorithm):
    class__ = "FMClassifier"
    module__ = "xenon_ext.nn"

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        estimator.fit(X, y, X_valid, y_valid)
        return estimator
