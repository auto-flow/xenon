import pandas as pd

from xenon.estimator.base import XenonEstimator
from xenon.hdl.hdl_constructor import HDL_Constructor
from xenon.tuner.tuner import Tuner

df = pd.read_csv("../data/QSAR.csv")

hdl_constructor = HDL_Constructor(
    DAG_workflow={
        "num->var": "compress.variance",
        "var->pea": {"_name": "compress.pearson", "n_jobs": 6},
        "pea->target": "logistic_regression"
    }
)
tuner = Tuner(
    run_limit=5,
    initial_runs=12,
    search_method="smac",
    n_jobs=1
)
xenon_pipeline = XenonEstimator(tuner, hdl_constructor)
column_descriptions = {
    "id": "Name",
    "target": "labels"
}

xenon_pipeline.fit(
    X_train=df, column_descriptions=column_descriptions
)
