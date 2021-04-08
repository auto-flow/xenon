#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
# from .meta import MetaDiscretizer
from xenon_ext.binning.decision_tree import get_decision_tree_binning_boundary
from xenon_ext.binning import DecisionTreeDiscretizer
from joblib import load

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    X_train,y_train=load("/data/1-4/data.bz2")
    transformer=DecisionTreeDiscretizer()
    X_train_bin=transformer.fit_transform(X_train,y_train)
    print(X_train_bin)

