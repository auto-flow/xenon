#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
# https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/preprocessing/_discretization.py#L21
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def get_columns(X):
    if isinstance(X, pd.DataFrame):
        columns = X.columns
    else:
        columns = range(X.shape[1])
    return columns


def get_col_vec(X, col):
    if isinstance(X, pd.DataFrame):
        vec = X[col].values
    else:
        vec = X[:, col]
    return vec


class MetaDiscretizer(BaseEstimator, TransformerMixin):
    binning_func = None

    def __init__(self):
        self.bin_edges_ = []

    def fit(self, X, y=None):
        columns = get_columns(X)
        # todo: 加n_jobs开并行
        for col in columns:
            vec = get_col_vec(X, col)
            self.bin_edges_.append(self.binning_func(vec, y))
        self.n_bins_ = [len(bin_edge) + 1 for bin_edge in self.bin_edges_]
        self._encoder = OneHotEncoder(
            categories=[np.arange(i) for i in self.n_bins_],
            sparse=True,
            dtype="int8"
        )
        self._encoder.fit(np.zeros((1, len(self.n_bins_))))
        return self

    def transform(self, X):
        bin_edges = self.bin_edges_
        columns = get_columns(X)
        Xt = np.zeros_like(X, dtype="int32")
        for i,(col, bin_edge) in enumerate(zip(columns, bin_edges)):
            vec = get_col_vec(X, col)
            discrete = np.digitize(vec, bin_edge)
            Xt[:, i] = discrete
        Xt = self._encoder.transform(Xt)
        # return Xt
        if isinstance(X, pd.DataFrame):
            ret_columns = []
            for i, col in enumerate(columns):
                for j in range(self.n_bins_[i]):
                    ret_columns.append(f"{col}-{j}")
            return pd.DataFrame.sparse.from_spmatrix(Xt, columns=ret_columns)
        else:
            return Xt
