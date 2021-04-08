#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
from .meta import MetaDiscretizer
from .decision_tree import get_decision_tree_binning_boundary
from .bayesian_block import get_bayesian_blocks_binning_boundary
from functools import partial


class DecisionTreeDiscretizer(MetaDiscretizer):
    def __init__(self, max_leaf_nodes=6, min_samples_leaf=0.05, min_max_bound=False):
        super(DecisionTreeDiscretizer, self).__init__()
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.binning_func = partial(
            get_decision_tree_binning_boundary,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
            min_max_bound=min_max_bound,
        )


class BayesianDiscretizer(MetaDiscretizer):
    def __init__(self):
        super(BayesianDiscretizer, self).__init__()
        self.binning_func = get_bayesian_blocks_binning_boundary
