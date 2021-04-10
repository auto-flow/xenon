#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
# from .meta import MetaDiscretizer
from xenon_ext.discretize.decision_tree import get_decision_tree_boundary
from xenon_ext.discretize import FlexibleDiscretizer
from joblib import load
from time import time

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.datasets import load_iris, load_digits, load_boston
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    discretizer = FlexibleDiscretizer(
        strategy="decision_tree",
        n_jobs=1, backend="multiprocessing")
    # X, y = load("/data/1-4/data.bz2")
    # y = y.values
    # X, y = load_digits(return_X_y=True)
    X, y = load_boston(return_X_y=True)
    # X = pd.DataFrame(X)
    # X['dummy'] = np.random.randint(0, 2, [150])
    lr = LogisticRegression(max_iter=100)
    start = time()
    Xt = discretizer.fit_transform(X, y)
    print(time() - start)
    pipeline = Pipeline([
        ("discretizer", discretizer),
        ("lr", lr),
    ])
    cv = StratifiedKFold(5, True, 42)
    scores = cross_val_score(pipeline, X, y, cv=cv)
    print(scores.mean())
