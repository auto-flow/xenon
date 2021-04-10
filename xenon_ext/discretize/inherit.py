#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.utils.multiclass import type_of_target

from .meta import MetaDiscretizer
from .decision_tree import get_decision_tree_boundary
from .bayesian_block import get_bayesian_blocks_boundary
from functools import partial


class DecisionTreeDiscretizer(MetaDiscretizer):
    def __init__(self, n_jobs=1, backend="threading", max_leaf_nodes=6, min_samples_leaf=0.05, min_max_bound=False):
        super(DecisionTreeDiscretizer, self).__init__(n_jobs, backend)
        self.min_max_bound = min_max_bound
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

    def fit(self, X, y=None):
        self.binning_func = partial(
            get_decision_tree_boundary,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            min_max_bound=self.min_max_bound,
            task="regression" if type_of_target(y) == "continuous" else "classification"
        )
        return super(DecisionTreeDiscretizer, self).fit(X, y)


class BayesianDiscretizer(MetaDiscretizer):
    def __init__(self):
        super(BayesianDiscretizer, self).__init__()
        self.binning_func = get_bayesian_blocks_boundary