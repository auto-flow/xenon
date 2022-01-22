#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import logging
import warnings
from copy import copy, deepcopy
from time import time

import lightgbm
import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder

from xenon_ext.occ_model.occ_lgbm import OCC_LGBMEstimator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GBDT_LR(OCC_LGBMEstimator):
    def __init__(
            self,
            n_estimators=100,  # [10, 150] 在测试中这个值越大越好
            objective=None,
            learning_rate=None,
            lr_its_multiply=1,  # [0.5 ~ 15]
            boosting_type="gbdt",
            # learning_rate=0.01,
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
            early_stopping_rounds=10,
            verbose=-1,
            n_jobs=-1,
            # -----------------------------------------------------------
            # lr params
            penalty="l2",
            # solver="saga",
            # l1_ratio=0.5,
            C=0.01,
            max_iter=100,
            iter_step=20,
            lr_es_round=2
    ):
        if learning_rate is None:
            learning_rate = lr_its_multiply / n_estimators
        super(GBDT_LR, self).__init__(
            n_estimators=n_estimators,
            objective=objective,
            boosting_type=boosting_type,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            random_state=random_state,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            subsample_for_bin=subsample_for_bin,
            min_child_weight=min_child_weight,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        self.C = C
        self.lr_es_round = lr_es_round
        self.iter_step = iter_step
        self.max_iter = max_iter
        self.penalty = penalty

    def _occ_train(self, i):
        ##############
        # GBDT Start #
        ##############
        gbt_start = time()
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
        leaf = self.pred_leaf(booster, self.train_path_list[i])
        #############
        # OHE Start #
        #############
        ohe_start = time()
        self.ohe = OneHotEncoder(
            dtype="int8",
            handle_unknown="ignore",
            # drop="first"
        ).fit(leaf)
        feature = self._transform_leaf(leaf)
        self.ohe_shape = feature.shape
        ############
        # LR Start #
        ############
        lr_start = time()
        can_es_lr = True
        leaf = self.pred_leaf(booster, self.valid_path_list[i])
        X_valid_transform = self._transform_leaf(leaf)
        self.performance_history = np.full(self.lr_es_round, -np.inf)
        self.iteration_history = np.full(self.lr_es_round, 0, dtype="int32")
        self.best_estimators = np.zeros([self.lr_es_round], dtype="object")
        if self.is_classification:
            lr = LogisticRegression(
                penalty=self.penalty,
                solver="lbfgs" if self.penalty == "l2" else "liblinear",
                C=self.C, random_state=0,
                # l1_ratio=self.l1_ratio,
                max_iter=self.iter_step
            )
        else:
            lr = Ridge(
                alpha=self.C, random_state=0, copy_X=False,
            )
        for i, max_iter in enumerate(range(self.iter_step, self.max_iter, self.iter_step)):
            lr.max_iter = max_iter
            lr.fit(feature, train_set.label)
            if can_es_lr:
                score = lr.score(X_valid_transform, valid_set.label)
                if np.any(score > self.performance_history):
                    index = i % self.lr_es_round
                    self.best_estimators[index] = deepcopy(lr)
                    self.performance_history[index] = score
                    self.iteration_history[index] = max_iter
                else:
                    break
        if can_es_lr:
            index = int(np.lexsort((self.iteration_history, -self.performance_history))[0])
            self.lr_best_iteration = int(self.iteration_history[index])
            lr = self.best_estimators[index]
            logger.info(f"{self.__class__.__name__} early_stopped, best_iteration_ = {self.lr_best_iteration}")
        else:
            self.lr_best_iteration = self.max_iter
        lr_end = time()
        logger.info(f"GBDT cost {ohe_start - gbt_start:.2f}s, OHE cost {lr_start - ohe_start:.2f}s, "
                    f"LR cost {lr_end - lr_start:.2f}s")
        self.labels.append(valid_set.label)
        copied_model = copy(self)
        copied_model.booster = booster
        copied_model.lr = lr
        return copied_model

    def pred_leaf(self, booster, dataset):
        leaf = booster.predict(dataset, num_iteration=booster.best_iteration, pred_leaf=True)
        return leaf

    def _transform_leaf(self, leaf):
        return self.ohe.transform(leaf)

    def predict_booster(self, dataset):
        leaf = self.pred_leaf(self.booster, dataset)
        ohe = self._transform_leaf(leaf)
        if self.is_classification:
            return self.lr.predict_proba(ohe)
        else:
            return self.lr.predict(ohe)


class GBDT_LR_Classifier(GBDT_LR, ClassifierMixin):
    is_classification = True


class GBDT_LR_Regressor(GBDT_LR, RegressorMixin):
    is_classification = False

    def predict_proba(self, X):
        raise NotImplementedError
