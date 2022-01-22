import os
from typing import List

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.metrics import r2_score, matthews_corrcoef
from sklearn.utils._testing import ignore_warnings

from xenon.ensemble.base import EnsembleEstimator
from xenon.lazy_import import fmin, tpe, hp, space_eval
from xenon.metrics import calculate_score, calculate_confusion_matrix
from xenon.utils import typing_
from xenon.utils.logging_ import get_logger


class StackEstimator(EnsembleEstimator):
    def __init__(
            self,
            meta_learner=None,
            use_features_in_secondary=False,
    ):
        self.use_features_in_secondary = use_features_in_secondary
        assert self.mainTask in ("classification", "regression")
        # todo: prepare parameter for user to extend choices of meta-learner and corresponding parameters
        # if not meta_learner:
        if self.mainTask == "classification":
            meta_cls = LogisticRegression
            meta_hps = dict(
                penalty='elasticnet',
                solver="saga",
                l1_ratio=hp.uniform('l1_ratio', 0, 1),
                C=hp.loguniform('C', np.log(0.01), np.log(10000)),  # anti human design
                fit_intercept=hp.choice('fit_intercept', [True, False]),  # fixme
                random_state=42
            )
        elif self.mainTask == "regression":
            meta_cls = ElasticNet
            meta_hps = dict(
                alpha=hp.loguniform('alpha', np.log(1e-2), np.log(10)),
                l1_ratio=hp.uniform('l1_ratio', 0, 1),
                fit_intercept=hp.choice('fit_intercept', [True, False]),  # fixme
                normalize=True,
                positive=True,  # force all coef_ to true
                random_state=42,
            )
        else:
            raise NotImplementedError
        self.meta_cls = meta_cls
        self.meta_hps = meta_hps
        self.logger = get_logger(self)

    @ignore_warnings(category=ConvergenceWarning)
    @ignore_warnings(category=FutureWarning)
    def fit_trained_data(
            self,
            estimators_list: List[List[typing_.GenericEstimator]],
            y_preds_list: List[List[np.ndarray]],
            y_true_indexes_list: List[List[np.ndarray]],
            y_true: np.ndarray
    ):
        super(StackEstimator, self).fit_trained_data(estimators_list, y_true_indexes_list, y_preds_list, y_true)
        meta_features = self.predict_meta_features(None, True)

        def objective(point):
            estimator = self.meta_cls(**point).fit(meta_features, self.stacked_y_true)
            if self.mainTask == "classification":
                y_pred = estimator.predict(meta_features)
                return 1 - matthews_corrcoef(self.stacked_y_true, y_pred)
            else:
                y_pred = estimator.predict(meta_features)
                return 1 - r2_score(self.stacked_y_true, y_pred)

        max_evals = int(os.getenv("AUTO_ENSEMBLE_TRIALS", 200))
        best = fmin(objective, self.meta_hps, algo=tpe.suggest, max_evals=max_evals, show_progressbar=False, verbose=0)
        best_point = space_eval(self.meta_hps, best)
        self.logger.info(f"meta_learner's hyper-parameters: ")
        for k, v in best_point.items():
            self.logger.info(f"\t{k}\t=\t{v}")
        self.meta_learner = self.meta_cls(**best_point)
        self.meta_learner.fit(meta_features, self.stacked_y_true)
        # fixme: 默认分类用mcc，回归用r2
        if self.mainTask == "classification":
            self.stacked_y_pred = self.meta_learner.predict_proba(meta_features)
            score = matthews_corrcoef(self.stacked_y_true, self.stacked_y_pred.argmax(axis=1))
        else:
            self.stacked_y_pred = self.meta_learner.predict(meta_features)
            score = r2_score(self.stacked_y_true, self.stacked_y_pred)
        _, self.all_score = calculate_score(
            self.stacked_y_true, self.stacked_y_pred, self.mainTask,
            should_calc_all_metric=True)
        # 除了算all_score，还要算混淆矩阵
        if self.mainTask == "classification":
            self.confusion_matrix = calculate_confusion_matrix(self.stacked_y_true, self.stacked_y_pred)
        else:
            self.confusion_matrix = None
        self.ensemble_score = score
        self.logger.info(f"meta_learner's performance : {score}")
        self.logger.info(f"meta_learner's coefficient : {self.meta_learner.coef_}")
        self.logger.info(f"meta_learner's intercept   : {self.meta_learner.intercept_}")
        coef_ = self.meta_learner.coef_
        if coef_.ndim == 2:
            self.weights = np.abs(coef_).sum(axis=1)
        elif coef_.ndim == 1:
            self.weights = coef_
        else:
            raise Exception
        self.meta_hps = None  # prevent pickle exception in tiny enviroment

    def predict_meta_features(self, X, is_train):
        raise NotImplementedError

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X, False)
        return predict_fn(meta_features)

    def predict(self, X):
        return self._do_predict(X, self.meta_learner.predict)
