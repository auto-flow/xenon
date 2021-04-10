#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-10
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, KernelPCA
import pandas as pd
from time import time
import logging

logger = logging.getLogger(__name__)


class FlexibleDecomposer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="none",
            n_components=0.9,
            kernel="poly",
            degree=2,
            gamma=0.1,
            coef0=0
    ):
        self.coef0 = coef0
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel
        self.n_components = n_components
        self.strategy = strategy
        assert strategy in ["none", "PCA", "KernelPCA"]

    def fit(self, X, y):
        if self.strategy == "PCA":
            self.decomposer = PCA(n_components=self.n_components)
        elif self.strategy == "KernelPCA":
            self.decomposer = KernelPCA(
                n_components=round(self.n_components * X.shape[0]),
                kernel=self.kernel,
                degree=self.degree,
                gamma=self.gamma,
                coef0=self.coef0
            )
        if self.strategy == "none":
            return self
        self.decomposer.fit(X, y)
        self.start_time = time()
        return self

    def transform(self, X):
        if self.strategy == "none":
            return X
        Xt = self.decomposer.transform(X)
        self.cost_time = time() - self.start_time
        logger.info(
            f"strategy = {self.strategy}, "
            f"decompose space from {X.shape[1]} dims to {Xt.shape[1]} components, "
            f"cost {self.cost_time:.3f} s.")
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                Xt,
                columns=[f"component-{i}" for i in range(Xt.shape[1])],
                index=X.index)
        else:
            return Xt
