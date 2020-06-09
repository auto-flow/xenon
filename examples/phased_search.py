import pandas as pd

from xenon.core.base import XenonEstimator
from xenon.hdl.hdl_constructor import HDL_Constructor
from xenon.tuner import Tuner

df_train = pd.read_csv("./data/train_classification.csv")

hdl_constructors = [
    HDL_Constructor(
        DAG_workflow={
            "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
            "lowR_nan->nan": "impute.fill_abnormal",
            "highR_nan->nan": "operate.drop",
            "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
            "cat->num": "encode.ordinal",
            "num->selected": [
                {"_name": "select.from_model_clf", "_select_percent": 80},
                {"_name": "select.rfe_clf", "_select_percent": 80},
            ],
            "selected->target": {"_name": "logistic_regression", "_vanilla": True}
        }
    ),
    HDL_Constructor(
        DAG_workflow={
            "nan->{highR=highR_nan,lowR=lowR_nan}": "operate.split.nan",
            "lowR_nan->nan": "impute.fill_abnormal",
            "highR_nan->nan": "operate.drop",
            "all->{cat_name=cat,num_name=num}": "operate.split.cat_num",
            "cat->num": "encode.ordinal",
            "num->selected": {"_name": "<mask>",
                         "_select_percent": {"_type": "quniform", "_value": [1, 100, 0.5],
                                             "_default": 80}},
            "selected->target": {"_name": "logistic_regression", "_vanilla": True}
        }
    ),
]

tuners = [
    Tuner(
        run_limit=-1,
        search_method="grid",
        n_jobs=3,
        debug=True
    ),
    Tuner(
        run_limit=50,
        initial_runs=10,
        search_method="smac",
        n_jobs=3,
        debug=True
    ),
]
xenon_pipeline = XenonEstimator(tuners, hdl_constructors)
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}

xenon_pipeline.fit(
    X_train=df_train, column_descriptions=column_descriptions
)
