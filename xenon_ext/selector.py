#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-09
# @Contact    : qichun.tang@bupt.edu.cn
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Xenon4bigdata_FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X=None, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        assert isinstance(X, pd.DataFrame), \
            ValueError('Xenon4bigdata_FeatureSelector只接受pandas.DataFrame对象，因为需要对列的信息进行匹配！！！')
        return X[self.columns]
