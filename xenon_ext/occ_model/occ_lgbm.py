#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import logging
import os
from copy import copy

import lightgbm
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.datasets import load_svmlight_file
from sklearn.utils import check_array
from sklearn.utils.multiclass import type_of_target

from xenon_ext.occ_model.base_occ import BaseOCC

logger = logging.getLogger(__name__)

TEST = os.getenv("TEST")


class OCC_LGBMEstimator(BaseOCC):

    def get_specific_boosting_type(self, boosting_type):
        return boosting_type

    def __init__(
            self,
            n_estimators=1000,
            objective=None,
            boosting_type="gbdt",
            learning_rate=0.01,
            max_depth=31,
            num_leaves=31,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=1,
            random_state=0,
            lambda_l1=0.1,
            lambda_l2=0.2,
            subsample_for_bin=40000,
            min_child_weight=0.01,
            early_stopping_rounds=100,
            verbose=-1,
            n_jobs=-1,
    ):
        super(OCC_LGBMEstimator, self).__init__()
        assert self.is_classification is not None, NotImplementedError
        self.n_jobs = n_jobs
        self.objective = objective
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.min_child_weight = min_child_weight
        self.subsample_for_bin = subsample_for_bin
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.random_state = random_state
        self.bagging_freq = bagging_freq
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.boosting_type = self.get_specific_boosting_type(boosting_type)
        self.n_estimators = n_estimators
        self.booster = None
        self.current_iterations = 0
        self.early_stopped = False
        if self.objective is None:
            if self.is_classification:
                if self.target_type == "binary":
                    self.objective = "binary"
                elif self.target_type == "multiclass":
                    self.objective = "multiclass"
                else:
                    raise ValueError(f"Invalid target_type {self.target_type}!")
            else:
                self.objective = "regression"
        params = dict(
            verbose=-1,
            boosting_type=self.boosting_type,
            objective=self.objective,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            random_state=self.random_state,
            lambda_l1=self.lambda_l1,
            lambda_l2=self.lambda_l2,
            subsample_for_bin=self.subsample_for_bin,
            min_child_weight=self.min_child_weight,
            num_threads=self.n_jobs
        )
        if self.objective == "multiclass":
            params.update({"num_class": self.n_classes})
        self.params = params

    def _occ_train(self, i):
        # 通过label判断objective
        train_set = lightgbm.Dataset(self.train_path_list[i])
        valid_set = lightgbm.Dataset(self.valid_path_list[i])
        booster = lightgbm.train(
            params=self.params,
            train_set=train_set,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_set],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose,
            # init_model=self.booster  # 不需要warm_start了
        )
        copied_model = copy(self)
        copied_model.booster = booster
        return copied_model

    def predict_booster(self, dataset):
        y_pred = self.booster.predict(dataset, num_iteration=self.booster.best_iteration)
        if self.objective == "binary" and y_pred.ndim == 1:
            y_pred = y_pred[:, None]
            y_pred = np.hstack([1 - y_pred, y_pred])
        return y_pred

    def _occ_validate(self, i, model):
        label = self.labels[i]
        y_pred = model.predict_booster(self.valid_path_list[i])
        if TEST:
            X, y = load_svmlight_file(self.valid_path_list[i])
            X = X.toarray()
            y_pred2 = model.predict_booster(X)
            assert np.all(y_pred == y_pred2)
            assert np.all(label == y)
        return label, y_pred

    def predict(self, X):
        X = check_array(X)
        if self.is_classification:
            return self.predict_proba(X).argmax(axis=1)
        else:
            return self.predict_booster(X)

    def predict_proba(self, X):
        X = check_array(X)
        return self.predict_booster(X)


class OCC_LGBMClassifier(OCC_LGBMEstimator, ClassifierMixin):
    is_classification = True


class OCC_LGBMRegressor(OCC_LGBMEstimator, RegressorMixin):
    is_classification = False

    def predict_proba(self, X):
        raise NotImplementedError


class OCC_RFClassifier(OCC_LGBMClassifier):
    def get_specific_boosting_type(self, boosting_type):
        return "rf"


class OCC_RFRegressor(OCC_LGBMRegressor):
    def get_specific_boosting_type(self, boosting_type):
        return "rf"
