#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from xenon.estimator.wrap_lightgbm import LGBMClassifier
import numpy as np

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
lgbm = LGBMClassifier(n_estimators=100, verbose=100)
lgbm.fit(X_train, y_train, X_test, y_test)
print(lgbm.score(X_test, y_test))
y_score = lgbm.predict_proba(X_test)
assert y_score.shape[1] == 10
# assert np.all(y_score.sum(axis=1) == 1)