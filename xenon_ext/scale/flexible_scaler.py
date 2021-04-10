#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-10
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

strategy_to_class = {
    "minmax": MinMaxScaler,
    "standardize": StandardScaler

}


class FlexibleScaler(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="minmax"
    ):
        self.strategy = strategy
        assert strategy in ["minmax", "standardize", "none"]
        self.scaler = None if strategy == "none" else strategy_to_class[strategy]()

    def fit(self, X, y):
        if self.strategy == "none":
            return self
        self.scaler.fit(X, y)
        return self

    def transform(self, X):
        if self.strategy == "none":
            return X
        Xt = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(Xt, columns=X.columns, index=X.index, dtype="float32")
        else:
            return Xt
