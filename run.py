#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-03-31
# @Contact    : qichun.tang@bupt.edu.cn
from xenon import XenonClassifier
from xenon import HDL_Constructor
from sklearn.datasets import load_digits

clf_workflow = {
    "num->selected": {
        "_name": "select.from_model_clf",
        "_vanilla": True,
        "estimator": {"_type": "choice", "_value":
            ["sklearn.ensemble.ExtraTreesClassifier", "sklearn.ensemble.RandomForestClassifier"],
                      "_default": "sklearn.ensemble.ExtraTreesClassifier"},
        "n_estimators": 10,
        "max_depth": 7,
        "min_samples_split": 10,
        "min_samples_leaf": 10,
        "random_state": 0,
        "n_jobs": 1,
        "_select_percent": {"_type": "quniform", "_value": [1, 80, 0.1], "_default": 40}
    },
    "selected->target": [
        # "adaboost",
        # "extra_trees",
        # "random_forest",
        "liblinear_svc",
        "libsvm_svc",
        # "lightgbm",
        # "logistic_regression"
    ],
}
hdl_constructor = HDL_Constructor(
    DAG_workflow=clf_workflow, balance_strategies=["weight"])
X, y = load_digits(return_X_y=True)
xenon = XenonClassifier(
    use_BOHB=True, n_jobs=1,
    hdl_constructor=hdl_constructor,
    per_run_memory_limit=30720,
    n_jobs_in_algorithm=3,
    imbalance_threshold=0.9
)
xenon.fit(X, y, )
print(xenon)
