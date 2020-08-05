#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy

import numpy as np
import pandas as pd
from joblib import load

from xenon.feature_engineer.encode.entity_encoder import EntityEncoder
from xenon.utils.logging_ import setup_logger

setup_logger()
X_train, y_train, X_test, y_test, cat = load("126025.bz2")
nan_cnt = np.count_nonzero(pd.isna(pd.concat([X_train, X_test])), axis=0)
X_train = X_train.loc[:, nan_cnt == 0]
X_test = X_test.loc[:, nan_cnt == 0]
cat = pd.Series(cat)
cat = cat[nan_cnt == 0]
cat = pd.Series(cat)
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder().fit(y_train)
y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)
X_train_copied = deepcopy(X_train)
cols=X_train.columns[cat].tolist()
entity_encoder = EntityEncoder(max_epoch=10,cols=cols).fit(X_train, y_train)
transformed = entity_encoder.transform(X_test)
print(transformed)
assert np.all(X_train == X_train_copied)
s = pd.Series(['age', 'fnlwgt', 'education_0', 'education_1', 'education_2',
               'education_3', 'education-num', 'marital-status_0', 'marital-status_1',
               'relationship_0', 'relationship_1', 'race_0', 'race_1', 'sex_0',
               'capital-gain', 'capital-loss', 'hours-per-week'])
assert np.all(s == transformed.columns)
assert np.all(transformed.index == X_test.index)
