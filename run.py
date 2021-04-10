#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-03-31
# @Contact    : qichun.tang@bupt.edu.cn
from xenon import XenonClassifier
from xenon import HDL_Constructor
from sklearn.datasets import load_digits
from joblib import load

clf_workflow = {
    "num->var_selected": "select.variance",
    "var_selected->scaled": "scale.flexible",
    "scaled->model_selected": "select.flexible",
    "model_selected->discretized": {
        # "_name": "decompose.flexible",
        "_name": "discretize.flexible",
        # "strategy":  {"_type": "choice",
        #   "_value": ["decision_tree", "none"],"_default": "none"},
        # "strategy": "decision_tree"
    },
    "discretized->target": [
        # "adaboost",
        # "extra_trees",
        "random_forest",
        # "liblinear_svc",
        "lightgbm",
        "logistic_regression",
        "gbdt_lr",
    ],
}
hdl_constructor = HDL_Constructor(
    DAG_workflow=clf_workflow)
X, y = load_digits(return_X_y=True)
# X, y = load("/data/1-4/data.bz2")
# y = y.values
xenon = XenonClassifier(
    use_xenon_opt=True, n_jobs=3,  # use_BOHB=True,
    hdl_constructor=hdl_constructor,
    per_run_memory_limit=30720,
    n_jobs_in_algorithm=4,
    imbalance_threshold=20,
    # total_time_limit=10
)
xenon.fit(X, y, )
print(xenon)
