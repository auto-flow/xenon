#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : qichun.tang@xtalpi.com
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import KFold

import xenon
from xenon import XenonClassifier

examples_path = Path(xenon.__file__).parent.parent / "examples"
train_df = pd.read_csv(examples_path / "data/train_classification.csv")
test_df = pd.read_csv(examples_path / "data/test_classification.csv")
trained_pipeline = XenonClassifier(
    initial_runs=1, run_limit=1, n_jobs=1,
    included_classifiers=["lightgbm"], debug=True,
    should_store_intermediate_result=True,
)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
# if not os.path.exists("xenon_classification.bz2"):
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42), fit_ensemble_params=False
)
joblib.dump(trained_pipeline, "xenon_classification.bz2")
predict_pipeline = joblib.load("xenon_classification.bz2")
result = predict_pipeline.predict(test_df)
print(result)
