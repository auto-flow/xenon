#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-07
# @Contact    : qichun.tang@bupt.edu.cn
# boruta算法： https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwj7mN7k2uvvAhV06nMBHTjgB4EQFjAAegQIBRAD&url=https%3A%2F%2Fwww.jstatsoft.org%2Fv36%2Fi11%2Fpaper%2F&usg=AOvVaw2qnt0VVYz0sfb7QqoUczmz
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import type_of_target
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd
import warnings

rf_classes = [RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor]


# get_support() 函数
# feature_importances_ 变量

class FlexibleFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="none",
            C=1,
            should_select_percent=False,
            select_percent=80,
            rf_n_estimators=20,
            gbdt_n_estimators=50,
            threshold=0,
            rf_type="random_forest",
            n_jobs=1
    ):
        self.n_jobs = n_jobs
        self.threshold = threshold
        assert strategy in [
            "none",
            "l1_linear",
            "rf",
            "gbdt",
        ]
        assert rf_type in [
            "random_forest",
            "extra_trees",
        ]
        self.rf_type = rf_type
        self.gbdt_n_estimators = gbdt_n_estimators
        self.rf_n_estimators = rf_n_estimators
        self.select_percent = select_percent
        self.should_select_percent = should_select_percent
        self.C = C
        self.strategy = strategy
        self.task = None

    def fit(self, X, y):
        if self.strategy == "none":
            return self
        if type_of_target(y) == "continuous":
            self.task = "regression"
        else:
            self.task = "classification"
        if self.strategy == "l1_linear":
            if self.task == "classification":
                self.model = LogisticRegression(
                    penalty="l1", C=self.C, solver="liblinear")
            else:
                self.model = Lasso(alpha=self.C)
        elif self.strategy == "rf":
            if self.rf_type == "random_forest":
                prefix = "RandomForest"
            else:
                prefix = "ExtraTrees"
            if self.task == "classification":
                suffix = "Classifier"
            else:
                suffix = "Regressor"
            self.model = eval(prefix + suffix)(
                n_estimators=self.rf_n_estimators,
                n_jobs=self.n_jobs
            )
        elif self.strategy == "gbdt":
            gbdt_params = dict(
                n_estimators=self.gbdt_n_estimators,
                importance_type="gain",
                n_jobs=self.n_jobs
            )
            if self.task == "classification":
                self.model = LGBMClassifier(**gbdt_params)
            else:
                self.model = LGBMRegressor(**gbdt_params)
        else:
            raise NotImplementedError
        self.model.fit(X, y)
        if self.strategy == "l1_linear":
            coef_ = self.model.coef_
            if coef_.ndim == 2:
                coef_ = coef_.sum(axis=0)
            self.feature_importances_ = np.abs(coef_)
        else:
            self.feature_importances_ = self.model.feature_importances_
        if self.should_select_percent:
            n_choose = round((self.select_percent / 100) * X.shape[1])
            n_choose = max(1, n_choose)
            n_choose = min(n_choose, X.shape[1])
            feat_idx = np.argsort(-self.feature_importances_)[:n_choose]
            self.support_ = np.zeros([X.shape[1]], dtype="bool")
            self.support_[feat_idx] = True
        else:
            self.support_ = (self.feature_importances_ > self.threshold)
            if not np.any(self.support_):
                warnings.warn("all feature_importances_ = 0")
                self.support_[np.random.randint(0, X.shape[1])] = True
        return self

    def get_support(self):
        if self.strategy == "none":
            raise NotImplementedError
        return self.support_

    def transform(self, X):
        if self.strategy == "none":
            return X
        if isinstance(X, pd.DataFrame):
            X = X.loc[:, self.support_]
        else:
            X = X[:, self.support_]
        return X
