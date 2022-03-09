#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from autoflow.hdl.utils import _get_default_hdl_bank
#dict_keys(['adaboost', 'bernoulli_nb', 'catboost', 'decision_tree',
# 'extra_trees', 'gbt_lr', 'gaussian_nb', 'gradient_boosting', 'knn',
# 'lda', 'linearsvc', 'svc', 'lightgbm', 'logistic_regression',
# 'multinomial_nb', 'qda', 'random_forest', 'sgd', 'tabular_nn'])
hdl_bank=_get_default_hdl_bank()
print(hdl_bank['classification']['knn'])
print(hdl_bank['classification']['svc'])