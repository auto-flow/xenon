import os
from typing import List

import numpy as np
from hyperopt import fmin, tpe
from hyperopt import hp, space_eval
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.utils._testing import ignore_warnings

from xenon.ensemble.base import EnsembleEstimator
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

    # def fit(self, X, y):
    #     # fixme: 2020-4-9 更新后， 此方法弃用
    #     # todo ： 验证所有的 y_true_indexes 合法
    #     meta_features = self.predict_meta_features(X, True)
    #     self.meta_learner.fit(meta_features, y)

    @ignore_warnings(category=ConvergenceWarning)
    @ignore_warnings(category=FutureWarning)
    def fit_trained_data(
            self,
            estimators_list: List[List[typing_.GenericEstimator]],
            y_preds_list: List[List[np.ndarray]],
            y_true_indexes_list: List[List[np.ndarray]],
            y_true: np.ndarray
    ):
        super(StackEstimator, self).fit_trained_data(estimators_list, y_preds_list, y_true_indexes_list, y_true)
        meta_features = self.predict_meta_features(None, True)

        def objective(point):
            return -self.meta_cls(**point).fit(meta_features, self.stacked_y_true). \
                score(meta_features, self.stacked_y_true)

        max_evals = int(os.getenv("AUTO_ENSEMBLE_TRIALS", 50))
        best = fmin(objective, self.meta_hps, algo=tpe.suggest, max_evals=max_evals, show_progressbar=False, verbose=0)
        best_point = space_eval(self.meta_hps, best)
        self.logger.info(f"meta_learner's hyper-parameters: ")
        for k, v in best_point.items():
            self.logger.info(f"\t{k}\t=\t{v}")
        self.meta_learner = self.meta_cls(**best_point)
        self.meta_learner.fit(meta_features, self.stacked_y_true)
        score = self.meta_learner.score(meta_features, self.stacked_y_true)  # it is a reward for bayesian model
        self.ensemble_score = score
        self.logger.info(f"meta_learner's performance : {score}")
        self.logger.info(f"meta_learner's coefficient : {self.meta_learner.coef_}")
        self.logger.info(f"meta_learner's intercept   : {self.meta_learner.intercept_}")

    def predict_meta_features(self, X, is_train):
        raise NotImplementedError

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X, False)
        return predict_fn(meta_features)

    def predict(self, X):
        return self._do_predict(X, self.meta_learner.predict)
