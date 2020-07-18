#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import numpy as np
import pandas as pd
from pathlib import  Path
from xenon import DataManager
from xenon.hdl.hdl_constructor import HDL_Constructor
import os

docs_dir = Path(__file__).parent.parent / "docs"

workflow = {
    "num->selected": {
        "_name": "select.from_model_clf",
        "_vanilla": True,
        "estimator": {"_type": "choice", "_value":
            ["sklearn.ensemble.ExtraTreesClassifier", "sklearn.ensemble.RandomForestClassifier"],
                      "_default": "sklearn.ensemble.ExtraTreesClassifier"},
        "n_estimators": 10,
        "max_depth": 7,
        "min_samples_split": 10,
        "min_samples_leaf": 10,
        "random_state": 0,
        "n_jobs": 1,
        "_select_percent": {"_type": "quniform", "_value": [1, 80, 0.1], "_default": 40}
    },
    "selected->target": [
        "adaboost",
        "extra_trees",
        "random_forest",
        "liblinear_svc",
        "libsvm_svc",
        "lightgbm",
        "logistic_regression"
    ]
}
hdl_constructor = HDL_Constructor(DAG_workflow=workflow)
data = pd.DataFrame(np.zeros([2, 2]), columns=["col", "target"])
column_descriptions = {
    "num": ["col"],
    "target": "target"
}
data_manager = DataManager(X_train=data, column_descriptions=column_descriptions)
hdl_constructor.run(data_manager)
# table view
df=hdl_constructor.get_hdl_dataframe()
html=df.to_html()
(docs_dir/"misc_table"/"demo_workflow.html").write_text(html)
graph = hdl_constructor.draw_workflow_space()
# graph view
gv_path=(docs_dir/"misc_table"/"demo_workflow.gv").as_posix()
png_path=(docs_dir/"misc_table"/"demo_workflow.png").as_posix()
open(gv_path, "w+").write(graph.source)
cmd = f'''dot -Tpng -Gsize=9,15\! -Gdpi=300 -o{png_path} {gv_path}'''
os.system(cmd)

