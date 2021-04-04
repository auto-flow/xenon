#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
# from .meta import MetaDiscretizer
from xenon_ext.binning.decision_tree import get_decision_tree_binning_boundary
from xenon_ext.binning import BayesianDiscretizer

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    discretizer = BayesianDiscretizer()
    discretizer.binning_func = get_decision_tree_binning_boundary
    X, y = load_iris(True)
    lr = LogisticRegression()
    pipeline = Pipeline([
        ("discretizer", discretizer),
        ("lr", lr),
    ])
    cv=StratifiedKFold(5,True,42)
    scores=cross_val_score(pipeline,X,y,cv=cv)
    print(scores.mean())
