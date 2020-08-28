#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

from xenon.estimator.wrap_lightgbm import LGBMClassifier as AFLGBMClassifier

X, y = load_iris(True)
af_lgbm = AFLGBMClassifier().fit(X, y)
lgbm = LGBMClassifier()
lgbm._n_features = af_lgbm.model.num_feature()
lgbm._Booster = af_lgbm.model
_n_classes = af_lgbm.model.num_model_per_iteration()
lgbm._n_classes = _n_classes
lgbm._le = LabelEncoder().fit(np.arange(_n_classes))
predictions = lgbm.predict(X)
print(predictions)
