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

hdl_constructor = HDL_Constructor(
    DAG_workflow={
        "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
        "lowR_nan->nan": "impute.fill_abnormal",
        "highR_nan->nan": "operate.drop",
        "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
        "cat->num": ["encode.cat_boost", "encode.target", "encode.label"],
        "num->target": {"_name": "lightgbm", "_vanilla": True}
    }
)
tuner = Tuner(
    run_limit=-1,
    search_method="grid"
)
xenon_pipeline = XenonEstimator(tuner, hdl_constructor)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

xenon_pipeline.fit(
    X_train=df_train, X_test=df_test, column_descriptions=column_descriptions
)
