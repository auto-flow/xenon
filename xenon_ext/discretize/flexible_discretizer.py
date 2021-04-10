#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-10
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.base import BaseEstimator, TransformerMixin
from .inherit import DecisionTreeDiscretizer


class FlexibleDiscretizer(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            strategy="none", n_jobs=1, backend="threading",
            max_leaf_nodes=6, min_samples_leaf=0.05
    ):
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.backend = backend
        self.n_jobs = n_jobs
        self.strategy = strategy
        assert strategy in ["decision_tree", "none"]
        if strategy == "none":
            self.discretizer = None
        elif strategy == "decision_tree":
            self.discretizer = DecisionTreeDiscretizer(
                n_jobs=n_jobs, backend=backend, max_leaf_nodes=max_leaf_nodes,
                min_samples_leaf=min_samples_leaf
            )
        else:
            raise NotImplementedError

    def fit(self, X, y):
        if self.strategy == "none":
            return self
        self.discretizer.fit(X, y)
        return self

    def transform(self, X):
        if self.strategy == "none":
            return X
        return self.discretizer.transform(X)
