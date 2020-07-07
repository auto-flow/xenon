#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from xenon import HDL_Constructor
from xenon.core.classifier import XenonClassifier
from xenon.ensemble.vote.classifier import VoteClassifier
from xenon.resource_manager.http import HttpResourceManager
from xenon.tuner import Tuner

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=30)
http_resource_manager = HttpResourceManager()
# http_resource_manager.login()
hdl_constructors = [
    HDL_Constructor(
        DAG_workflow={
            "num->target": ["liblinear_svc", "libsvm_svc", "logistic_regression"]
        },
    )
]
tuners = [
    # Tuner(
    #     search_method="random",
    #     run_limit=3,
    #     n_jobs=3,
    #     debug=True
    # ),
    Tuner(
        search_method="smac",
        initial_runs=3,
        run_limit=6,
        n_jobs=3,
        debug=True
    )
]
pipe = XenonClassifier(
    hdl_constructor=hdl_constructors,
    tuner=tuners,
    resource_manager=http_resource_manager
)
pipe.fit(
    X_train, y_train,
    fit_ensemble_params={"trials_fetcher_params":{"k":20}},

)
# assert isinstance(pipe.estimator, VoteClassifier)
# score = accuracy_score(y_test, y_pred)
score = pipe.score(X_test, y_test)
print(score)
