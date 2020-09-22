#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from xenon.ensemble.mean.regressor import MeanRegressor
from xenon.ensemble.vote.classifier import VoteClassifier
from xenon.utils.ml_task import MLTask


def ensemble_folds_estimators(estimators, ml_task: MLTask):
    # len(estimators) == n_folds
    if ml_task.mainTask == "classification":
        ensemble_estimator = VoteClassifier(estimators)
    else:
        ensemble_estimator = MeanRegressor(estimators)
    return ensemble_estimator
