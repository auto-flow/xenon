#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder

from xenon.datasets import load_task
from xenon.estimator.tabular_nn_est import TabularNNClassifier
from xenon.utils.logging_ import setup_logger

setup_logger()

class RunTabularNN():
    current_file = __file__

    def __init__(self):
        cur_dir = Path(self.current_file).parent
        if (cur_dir / "126025.bz2").exists():
            X_train, y_train, X_test, y_test, cat = joblib.load(cur_dir / "126025.bz2")
        else:
            X_train, y_train, X_test, y_test, cat = load_task(126025)
            joblib.dump(
                [X_train, y_train, X_test, y_test, cat],
                cur_dir / "126025.bz2"
            )
        nan_cnt = np.count_nonzero(pd.isna(pd.concat([X_train, X_test])), axis=0)
        cat = np.array(cat)
        cat_na_mask = (nan_cnt > 0) & cat
        num_na_mask = (nan_cnt > 0) & (~cat)
        cat_imputer = SimpleImputer(strategy="constant", fill_value="NA").fit(X_train.loc[:, cat_na_mask])
        # num_imputer = SimpleImputer(strategy="median").fit(X_train.loc[:, num_na_mask])
        X_train.loc[:, cat_na_mask] = cat_imputer.transform(X_train.loc[:, cat_na_mask])
        X_test.loc[:, cat_na_mask] = cat_imputer.transform(X_test.loc[:, cat_na_mask])
        # X_train.loc[:, num_na_mask] = num_imputer.transform(X_train.loc[:, num_na_mask])
        # X_test.loc[:, num_na_mask] = num_imputer.transform(X_test.loc[:, num_na_mask])
        ordinal_encoder = OrdinalEncoder(dtype="int").fit(X_train.loc[:, cat])
        transformer = StandardScaler().fit(X_train.loc[:, ~cat])
        X_train.loc[:, cat] = ordinal_encoder.transform(X_train.loc[:, cat])
        X_train.loc[:, ~cat] = transformer.transform(X_train.loc[:, ~cat])
        X_test.loc[:, cat] = ordinal_encoder.transform(X_test.loc[:, cat])
        X_test.loc[:, ~cat] = transformer.transform(X_test.loc[:, ~cat])
        self.cat_indexes = np.arange(len(cat))[cat]
        label_encoder = LabelEncoder().fit(y_train)
        self.y_train = label_encoder.transform(y_train)
        self.y_test = label_encoder.transform(y_test)
        self.X_train = X_train
        self.X_test = X_test

    def run_adult_dataset(self):
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1,  # , class_weight="balanced"
        )
        tabular.fit(self.X_train, self.y_train, self.X_test, self.y_test, categorical_feature=self.cat_indexes.tolist())
        print(tabular.score(self.X_test, self.y_test))

    def run_multiclass(self):
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        tabular = TabularNNClassifier(
            verbose=1, max_epoch=64, early_stopping_rounds=16, n_jobs=-1, lr=1e-2
        )
        tabular.fit(X_train, y_train, X_test, y_test)
        print(tabular.score(X_test, y_test))
        y_score = tabular.predict_proba(X_test)
        assert y_score.shape[1] == 10
        assert np.all(np.abs(y_score.sum(axis=1) - 1) < 1e3)
        if tabular.early_stopped:
            assert tabular.best_estimators is None

RunTabularNN().run_multiclass()