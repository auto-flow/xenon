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
from joblib import delayed, Parallel
from xenon_ext.utils import get_chunks, parse_n_jobs
from time import time
from logging import getLogger

logger = getLogger(__name__)


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


def apply_binning_func_in_columns(X, y, columns, binning_func):
    bin_edges_ = []
    for col in columns:
        vec = get_col_vec(X, col)
        bin_edges_.append(binning_func(vec, y))
    return bin_edges_


class MetaDiscretizer(BaseEstimator, TransformerMixin):
    binning_func = None

    def __init__(self, n_jobs=1, backend="threading"):
        self.backend = backend
        self.n_jobs = parse_n_jobs(n_jobs)
        self.bin_edges_ = []

    def fit(self, X, y=None):
        start_time = time()
        columns = get_columns(X)
        #  加n_jobs开并行
        columns_list = get_chunks(columns, self.n_jobs)
        bin_edges_list = Parallel(n_jobs=self.n_jobs, backend=self.backend)(
            delayed(apply_binning_func_in_columns)(X, y, columns_, self.binning_func)
            for columns_ in columns_list
        )
        for bin_edges in bin_edges_list:
            self.bin_edges_.extend(bin_edges)
        self.n_bins_ = [len(bin_edge) + 1 for bin_edge in self.bin_edges_]
        self._encoder = OneHotEncoder(
            categories=[np.arange(i) for i in self.n_bins_],
            sparse=True,
            dtype="int8",
            drop="first"
        )
        self._encoder.fit(np.zeros((1, len(self.n_bins_))))
        cost_time = time() - start_time
        logger.info(f"{self.__class__.__name__} cost {cost_time:.3f} s, "
                    f"from {X.shape[1]} features to {sum(self.n_bins_)} bins.")
        return self

    def transform(self, X):
        bin_edges = self.bin_edges_
        columns = get_columns(X)
        Xt = np.zeros_like(X, dtype="int32")
        for i, (col, bin_edge) in enumerate(zip(columns, bin_edges)):
            vec = get_col_vec(X, col)
            discrete = np.digitize(vec, bin_edge)
            Xt[:, i] = discrete
        Xt = self._encoder.transform(Xt)
        # return Xt
        if isinstance(X, pd.DataFrame):
            ret_columns = []
            for i, col in enumerate(columns):
                for j in range(self.n_bins_[i] - 1):
                    ret_columns.append(f"{col}-{j}")
            return pd.DataFrame.sparse.from_spmatrix(Xt, columns=ret_columns)
        else:
            return Xt
