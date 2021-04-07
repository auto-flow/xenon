#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
# from .meta import MetaDiscretizer
from xenon_ext.feature_selection import FlexibleFeatureSelector

if __name__ == '__main__':
    import pandas as pd
    from sklearn.datasets import load_iris, load_boston
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    selector=FlexibleFeatureSelector(
        strategy="rf",
        rf_type="extra_trees",
        should_select_percent=False
        # select_percent=50
    )
    X, y = load_boston(True)
    X = pd.DataFrame(X)
    Xt=selector.fit_transform(X,y)
    print(Xt)
