#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-14
# @Contact    : qichun.tang@bupt.edu.cn
import os
from pathlib import Path
from time import time

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import roc_auc_score

import xenon
from xenon_ext.occ_model.xlearn import XlearnClassifier

root = Path(xenon.__file__).parent.parent.as_posix()
start_time = time()
datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_clf'
os.environ['DATAPATH'] = datapath
lgbm = XlearnClassifier(
    model_type="lr",
    is_instance_norm=False,
    epoch=100,
    lr=0.01,
    stop_window=10,
    opt='ftrl',
    alpha=0.01
)
models = lgbm.occ_train()
labels, predictions = lgbm.occ_validate(models)
scores = []
scores2 = []
for i, (label, prediction) in enumerate(zip(labels, predictions)):
    df, label2 = load_svmlight_file(f'{datapath}/valid_{i}.txt')
    df = df.todense()
    pred2 = models[i].predict_proba(df)
    auc = roc_auc_score(label, prediction[:, 1])
    auc2 = roc_auc_score(label2, pred2[:, 1])
    scores.append(auc)
    scores2.append(auc2)
score = np.mean(scores)
score2 = np.mean(scores2)
cost_time = time() - start_time
print(score)
print(score2)
# self.assertEqual(score, score2)
print(cost_time)
from joblib import dump
model=models[0]
dump(model, "test_dump_model.pkl")
