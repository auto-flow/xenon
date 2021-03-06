import inspect
import math
from copy import deepcopy
from importlib import import_module
from time import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from xenon.data_container import DataFrameContainer
from xenon.data_container import NdArrayContainer
from xenon.data_container.base import DataContainer
from xenon.utils.logging_ import get_logger


class XenonComponent(BaseEstimator):
    module__ = None
    class__ = None
    classification_only = False
    regression_only = False
    boost_model = False
    tree_model = False
    cache_intermediate = False
    support_early_stopping = False
    is_fit = False

    def __init__(self, **kwargs):
        self.resource_manager = None
        self.component = None
        self.in_feature_groups = None
        self.out_feature_groups = None
        self.hyperparams = kwargs
        self.set_addition_info(kwargs)
        self.logger = get_logger(self)

    def _get_param_names(cls):
        return sorted(cls.hyperparams.keys())

    @property
    def class_(self):
        if not self.class__:
            raise NotImplementedError()
        return self.class__

    @property
    def module_(self):
        if not self.module__:
            raise NotImplementedError()
        return self.module__

    def get_estimator_class(self):
        M = import_module(self.module_)
        return getattr(M, self.class_)

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams = deepcopy(hyperparams)
        hyperparams = self.before_parse_escape_hyperparameters(hyperparams)
        should_pop = []
        updated = {}
        for key, value in hyperparams.items():
            key: str
            if key.startswith("_") and (not key.startswith("__")):
                should_pop.append(key)
                key = key[1:]
                new_key, indicator = key.split("__")
                updated[new_key] = self.parse_escape_hyperparameters(indicator, hyperparams, value)
        for key in should_pop:
            hyperparams.pop(key)
        hyperparams.update(updated)
        return hyperparams

    def after_process_estimator(self, estimator, X_train, y_train=None, X_valid=None, y_valid=None, X_test=None,
                                y_test=None):
        return estimator

    def before_fit_X(self, X: DataFrameContainer):
        if X is None:
            return None
        if isinstance(X, DataContainer):
            return X.data
        return X

    def before_fit_y(self, y: NdArrayContainer):
        if y is None:
            return None
        return y.data

    def filter_invalid(self, cls, hyperparams: Dict) -> Dict:
        hyperparams = deepcopy(hyperparams)
        validated = {}
        for key, value in hyperparams.items():
            if key in inspect.signature(cls.__init__).parameters.keys():
                validated[key] = value
            else:
                pass
        return validated

    def filter_feature_groups(self, X: Optional[DataFrameContainer]):
        if X is None:
            return None
        assert isinstance(X, DataFrameContainer)
        from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm
        if issubclass(self.__class__, XenonFeatureEngineerAlgorithm):
            df = X.filter_feature_groups(self.in_feature_groups)
        else:
            df = X
        # rectify_dtypes(df)
        return df

    def build_proxy_estimator(self):
        if self.component is not None:
            return
        # ?????????????????????????????????????????????????????????????????????????????????
        cls = self.get_estimator_class()
        # ???????????????????????????????????????
        self.processed_params = self.filter_invalid(
            cls, self.after_process_hyperparams(self.hyperparams)
        )
        self.component = cls(
            **self.processed_params
        )

    def fit(self, X_train, y_train=None,
            X_valid=None, y_valid=None,
            X_test=None, y_test=None):
        # ????????????????????????feature_groups
        assert isinstance(X_train, DataFrameContainer)
        X_train = self.filter_feature_groups(X_train)
        X_valid = self.filter_feature_groups(X_valid)
        X_test = self.filter_feature_groups(X_test)
        self.shape = X_train.shape
        self.build_proxy_estimator()
        feature_groups = X_train.feature_groups
        # ???????????????????????????????????????preprocessor????????????X>0????????????
        X_train_ = self.before_fit_X(X_train)
        y_train_ = self.before_fit_y(y_train)
        X_test_ = self.before_fit_X(X_test)
        y_test_ = self.before_fit_y(y_test)
        X_valid_ = self.before_fit_X(X_valid)
        y_valid_ = self.before_fit_y(y_valid)
        # ????????????estimator???????????????
        self.component = self.after_process_estimator(self.component, X_train_, y_train_, X_valid_,
                                                      y_valid_, X_test_, y_test_)
        # todo: ?????????????????????????????????
        if len(X_train.shape) > 1 and X_train.shape[1] > 0:
            self.component = self._fit(self.component, X_train_, y_train_, X_valid_,
                                       y_valid_, X_test_, y_test_, feature_groups)
            self.is_fit = True
        else:
            self.logger.warning(
                f"Component: {self.__class__.__name__} is fitting a empty data.\nShape of X_train_ = {X_train.shape}.")
        return self

    def prepare_X_to_fit(self, X_train, X_valid=None, X_test=None, **kwargs) -> pd.DataFrame:
        return self.before_fit_X(X_train)

    def _fit(self, estimator, X_train, y_train=None, X_valid=None,
             y_valid=None, X_test=None, y_test=None, feature_groups=None):
        # ???????????????????????????????????????????????????
        X = self.prepare_X_to_fit(X_train, X_valid, X_test)
        fitted_estimator = self.core_fit(estimator, X, y_train, X_valid, y_valid, X_test, y_test, feature_groups)
        return fitted_estimator

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        return estimator.fit(X, y)

    def set_addition_info(self, dict_: dict):
        for key, value in dict_.items():
            setattr(self, key, value)

    def get_estimator(self):
        return self.component

    def before_parse_escape_hyperparameters(self, hyperparams):
        return hyperparams

    def parse_escape_hyperparameters(self, indicator, hyperparams, value):
        if indicator == "lr_ratio":
            lr = hyperparams["learning_rate"]
            return max(int(value * (1 / lr)), 10)
        elif indicator == "sp1_ratio":
            factor = "shape"
            if hasattr(self, factor):
                n_components = max(1, min(self.shape[0], round(self.shape[1] * value)))
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 100
            return n_components
        elif indicator == "sp1_percent":
            factor = "shape"
            if hasattr(self, factor):
                n_components = max(
                    int(self.shape[1] * (value / 100)),
                    1
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 100
            return n_components
        elif indicator == "sp1_dev":
            factor = "shape"
            if hasattr(self, factor):
                if value == 0:
                    value = 1
                n_components = max(
                    math.ceil(self.shape[1] / value),
                    1
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 100
            return n_components
        elif indicator == "card_ratio":
            factor = "cardinality"
            if hasattr(self, factor):
                n_components = max(
                    math.ceil(self.cardinality * value),
                    2
                )
            else:
                self.logger.warning(f"{str(self)} haven't attribute {factor}")
                n_components = 6
            return n_components
        else:
            raise NotImplementedError()

    def before_pred_X(self, X: DataFrameContainer):
        return X.data


class XenonIterComponent(XenonComponent):
    warm_start_name = "warm_start"
    iterations_name = "n_estimators"
    support_early_stopping = True

    @ignore_warnings(category=ConvergenceWarning)
    def iterative_fit(self, X, y, X_valid, y_valid, iter_inc):
        s = time()
        self.component.fit(X, y)
        self.fit_times += time() - s
        early_stopping_tol = getattr(self, "early_stopping_tol", 0.001)
        N = len(self.performance_history)
        if X_valid is not None and y_valid is not None:
            s = time()
            test_performance = self.component.score(X_valid, y_valid)
            train_performance = self.component.score(X, y)
            self.score_times += time() - s
            self.learning_curve[0].append(self.iteration)
            self.learning_curve[1].append(train_performance)
            self.learning_curve[2].append(test_performance)
            self.learning_curve[3].append(self.fit_times)
            self.learning_curve[4].append(self.score_times)
            if np.any(test_performance - early_stopping_tol > self.performance_history):
                index = self.iter_ix % N
                self.best_estimators[index] = deepcopy(self.component)
                self.performance_history[index] = test_performance
                self.iteration_history[index] = self.iteration
            else:
                self.fully_fitted_ = True
                # todo: choose maximal performance, minimal iteration
                index = int(np.lexsort((self.iteration_history, -self.performance_history))[0])
                self.best_iteration_ = self.iteration_history[index]
                setattr(self, self.iterations_name, self.best_iteration_)
                # index = np.argmax(self.performance_history)
                best_estimator = self.best_estimators[index]
                self.best_estimators = None
                self.component = best_estimator
        if not self.fully_fitted:
            self.iteration_ = getattr(self.component, self.iterations_name)
            self.iteration_ += iter_inc
            self.iteration_ = min(self.iteration_, self.max_iterations)
            setattr(self.component, self.iterations_name, self.iteration_)

    def core_fit(self, estimator, X, y, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        # ??????????????????????????????????????????
        iter_inc = getattr(self, "iter_inc", 10)
        early_stopping_rounds = getattr(self, "early_stopping_rounds", 20)
        self.performance_history = np.full(early_stopping_rounds, -np.inf)
        self.iteration_history = np.full(early_stopping_rounds, -np.inf)
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve
        self.fit_times = 0
        self.score_times = 0
        self.learning_curve = [
            [],  # train_sizes_abs [0]
            [],  # train_scores    [1]
            [],  # test_scores     [2]
            [],  # fit_times       [3]
            [],  # score_times     [4]
        ]
        N = len(self.performance_history)
        self.best_estimators = np.zeros([N], dtype="object")

        self.iter_ix = 0
        while not self.fully_fitted:
            self.iterative_fit(X, y, X_valid, y_valid, iter_inc)
            self.iter_ix += 1
        return self.component

    @property
    def fully_fitted(self) -> bool:
        if self.iteration > self.max_iterations:
            return True
        return getattr(self, "fully_fitted_", False)

    @property
    def max_iterations(self):
        return getattr(self, "max_iterations_", 1000)

    @property
    def iteration(self):
        return getattr(self, "iteration_", 1)

    def after_process_hyperparams(self, hyperparams) -> Dict:
        iter_inc = getattr(self, "iter_inc", 10)
        hyperparams = super(XenonIterComponent, self).after_process_hyperparams(hyperparams)
        hyperparams[self.warm_start_name] = True
        if self.iterations_name in hyperparams:
            self.max_iterations_ = hyperparams[self.iterations_name]
        else:
            self.max_iterations_ = 1000
        hyperparams[self.iterations_name] = iter_inc
        self.iteration_ = iter_inc
        return hyperparams
