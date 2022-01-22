#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-08
# @Contact    : qichun.tang@bupt.edu.cn
import os
from pathlib import Path
from time import time
from unittest import TestCase

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score

import xenon
from xenon_ext.occ_model.gbdt_lr import GBDT_LR_Classifier, GBDT_LR_Regressor
from xenon_ext.occ_model.occ_lgbm import OCC_LGBMClassifier, OCC_LGBMRegressor
from xenon_ext.occ_model.occ_lgbm import OCC_RFClassifier, OCC_RFRegressor
from xenon_ext.occ_model.xlearn import XlearnClassifier, XlearnRegressor


class Test_OCC(TestCase):
    def test_lgbm_clf(self):
        root = Path(xenon.__file__).parent.parent.as_posix()

        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_clf'
        os.environ['DATAPATH'] = datapath
        lgbm = OCC_LGBMClassifier(n_estimators=1000, early_stopping_rounds=10)
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
        self.assertEqual(score, score2)
        print(cost_time)

    def test_lgbm_reg(self):
        root = Path(xenon.__file__).parent.parent.as_posix()

        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_reg'
        os.environ['DATAPATH'] = datapath
        lgbm = OCC_LGBMRegressor(
            boosting_type="gbdt",
            n_estimators=100, early_stopping_rounds=10, learning_rate=0.01)
        models = lgbm.occ_train()
        labels, predictions = lgbm.occ_validate(models)
        scores = []
        scores2 = []
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            df, label2 = load_svmlight_file(f'{datapath}/valid_{i}.txt')
            df = df.todense()
            pred2 = models[i].predict(df)
            auc = r2_score(label, prediction)
            auc2 = r2_score(label2, pred2)
            scores.append(auc)
            scores2.append(auc2)
        score = np.mean(scores)
        score2 = np.mean(scores2)
        cost_time = time() - start_time
        print(score)
        print(score2)
        # self.assertEqual(score, score2)
        print(cost_time)

    def test_rf_clf(self):
        root = Path(xenon.__file__).parent.parent.as_posix()

        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_clf'
        os.environ['DATAPATH'] = datapath
        lgbm = OCC_RFClassifier(n_estimators=1000, early_stopping_rounds=10)
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
        self.assertEqual(score, score2)
        print(cost_time)

    def test_rf_reg(self):
        root = Path(xenon.__file__).parent.parent.as_posix()

        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_reg'
        os.environ['DATAPATH'] = datapath
        lgbm = OCC_RFRegressor(
            boosting_type="gbdt",
            n_estimators=100, early_stopping_rounds=10, learning_rate=10000)
        models = lgbm.occ_train()
        labels, predictions = lgbm.occ_validate(models)
        scores = []
        scores2 = []
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            df, label2 = load_svmlight_file(f'{datapath}/valid_{i}.txt')
            df = df.todense()
            pred2 = models[i].predict(df)
            auc = r2_score(label, prediction)
            auc2 = r2_score(label2, pred2)
            scores.append(auc)
            scores2.append(auc2)
        score = np.mean(scores)
        score2 = np.mean(scores2)
        cost_time = time() - start_time
        print(score)
        print(score2)
        # self.assertEqual(score, score2)
        print(cost_time)

    def test_gbdt_lr_clf(self):
        root = Path(xenon.__file__).parent.parent.as_posix()
        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_clf'
        os.environ['DATAPATH'] = datapath
        lgbm = GBDT_LR_Classifier(n_estimators=150, early_stopping_rounds=10, lr_its_multiply=1)
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
        self.assertEqual(score, score2)
        print(cost_time)

    def test_gbdt_lr_reg(self):
        root = Path(xenon.__file__).parent.parent.as_posix()

        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_reg'
        os.environ['DATAPATH'] = datapath
        lgbm = GBDT_LR_Regressor(n_estimators=150, early_stopping_rounds=10, lr_its_multiply=1)
        models = lgbm.occ_train()
        labels, predictions = lgbm.occ_validate(models)
        scores = []
        scores2 = []
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            df, label2 = load_svmlight_file(f'{datapath}/valid_{i}.txt')
            df = df.todense()
            pred2 = models[i].predict(df)
            auc = r2_score(label, prediction)
            auc2 = r2_score(label2, pred2)
            scores.append(auc)
            scores2.append(auc2)
        score = np.mean(scores)
        score2 = np.mean(scores2)
        cost_time = time() - start_time
        print(score)
        print(score2)
        # self.assertEqual(score, score2)
        print(cost_time)

    def test_xlearn_clf(self):
        root = Path(xenon.__file__).parent.parent.as_posix()
        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_clf'
        os.environ['DATAPATH'] = datapath
        lgbm = XlearnClassifier(model_type="lr")
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
        self.assertEqual(score, score2)
        print(cost_time)

    def test_xlearn_reg(self):
        root = Path(xenon.__file__).parent.parent.as_posix()

        start_time = time()
        datapath = root + '/savedpath/xenon4bigdata/bd1_preprocess_reg'
        os.environ['DATAPATH'] = datapath
        lgbm = XlearnRegressor(
            model_type="lr",

        )
        models = lgbm.occ_train()
        labels, predictions = lgbm.occ_validate(models)
        scores = []
        scores2 = []
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            df, label2 = load_svmlight_file(f'{datapath}/valid_{i}.txt')
            df = df.todense()
            pred2 = models[i].predict(df)
            auc = r2_score(label, prediction)
            auc2 = r2_score(label2, pred2)
            scores.append(auc)
            scores2.append(auc2)
        score = np.mean(scores)
        score2 = np.mean(scores2)
        cost_time = time() - start_time
        print(score)
        print(score2)
        # self.assertEqual(score, score2)
        print(cost_time)
