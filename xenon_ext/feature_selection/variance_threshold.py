#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-07
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# get_support() 函数
# feature_importances_ 变量

class VarianceThreshold(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            threshold=0,
    ):
        self.threshold = threshold

    def fit(self, X, y=None):
        self.feature_importances_ = np.array(np.nanvar(X, axis=0))
        self.support_ = (self.feature_importances_ > self.threshold)
        logger.info(f"VarianceThreshold(threshold={self.threshold:.3f})"
                    f" delete {np.count_nonzero(~self.support_)} features. ")
        return self

    def get_support(self):
        return self.support_

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.loc[:, self.support_]
        else:
            X = X[:, self.support_]
        return X
