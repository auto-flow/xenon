import pandas as pd
from sklearn.model_selection import ShuffleSplit

from xenon.estimator.base import XenonEstimator
from xenon.hdl.hdl_constructor import HDL_Constructor
from xenon.tuner.tuner import Tuner

df = pd.read_csv("../examples/classification/train_classification.csv")
ss = ShuffleSplit(n_splits=1, random_state=0, test_size=0.25)
train_ix, test_ix = next(ss.split(df))
df_train = df.iloc[train_ix, :]
df_test = df.iloc[test_ix, :]

tuner = Tuner(
    initial_runs=5,
    run_limit=12,
)
hdl_constructor=HDL_Constructor(
    DAG_workflow={
                "nan->imp": "impute.fill_abnormal",
                "imp->{cat_name=cat,num_name=num}": "operate.split.cat_num",
                "cat->num": [
                    "encode.cat_boost",
                    {"_name":"encode.label","__proba":0.8},
                ],
                "num->num":["scale.normalize",{"_name":"None","__proba":0.8}],
                "num->target": {"_name":"catboost","_vanilla":False}
            }
)
xenon_pipeline = XenonEstimator(tuner, hdl_constructor)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

xenon_pipeline.fit(
    X_train=df_train, X_test=df_test, column_descriptions=column_descriptions, n_jobs=5
)
