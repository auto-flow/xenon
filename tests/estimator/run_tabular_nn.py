#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from xenon.estimator.tabular_nn_est import TabularNNClassifier
from xenon.utils.logging_ import setup_logger

setup_logger()

# X, y = load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# tabular = TabularNNRegressor(verbose=-1, epoch=64, early_stopping_rounds=32, n_jobs=3)
# tabular.fit(X_train, y_train, X_test, y_test)
# score = tabular.score(X_test, y_test)
# print(score)

from sklearn.datasets import load_iris
# X, y = load_iris(return_X_y=True)
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
tabular = TabularNNClassifier(
    verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1, batch_size=1024
)
tabular.fit(X_train, y_train)
score = tabular.score(X_test, y_test)
print(score)
