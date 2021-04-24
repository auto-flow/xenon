#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-21
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.datasets import load_iris,load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xenon_ext.nn.fm import FMClassifier, FMRegressor

def test_fm_ftrl_clf():
    X, y = load_iris(True)
    X = MinMaxScaler().fit_transform(X)
    y[y != 0] = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    fm = FMClassifier(
        use_fm=True, max_epoch=10000, weight_decay=0, optimizer="ftrl", lr=0.1,
        device="cpu", tol=1e-2
        # class_weight="balanced"
    )
    fm.fit(X_train, y_train, X_test, y_test)
    y_proba = fm.predict_proba(X_test)
    print(y_proba)
    print(fm.score(X_test, y_test))
    print(fm.fm_nn.lr.weight.data)
    try:
        print(fm.fm_nn.embd.data)
    except:
        pass

def test_fm_sgd_clf():
    X, y = load_iris(True)
    X = MinMaxScaler().fit_transform(X)
    y[y != 0] = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    fm = FMClassifier(
        use_fm=True, max_epoch=10000, weight_decay=0, optimizer="sgd", lr=0.1,
        device="cpu", tol=1e-2
        # class_weight="balanced"
    )
    fm.fit(X_train, y_train, X_test, y_test)
    y_proba = fm.predict_proba(X_test)
    print(y_proba)
    print(fm.score(X_test, y_test))
    print(fm.fm_nn.lr.weight.data)
    try:
        print(fm.fm_nn.embd.data)
    except:
        pass

def test_fm_ftrl_reg():
    X, y = load_boston(True)
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    fm = FMRegressor(
        use_fm=True, max_epoch=10000, weight_decay=0, optimizer="ftrl", lr=0.01,
        device="cpu", tol=1e-3
        # class_weight="balanced"
    )
    fm.fit(X_train, y_train, X_test, y_test)
    print(fm.score(X_test, y_test))
    print(fm.fm_nn.lr.weight.data)
    try:
        print(fm.fm_nn.embd.data)
    except:
        pass

def test_fm_sgd_reg():
    X, y = load_boston(True)
    X = MinMaxScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    fm = FMRegressor(
        use_fm=True, max_epoch=10000, weight_decay=0, optimizer="sgd", lr=0.01,
        device="cpu", tol=1e-3
        # class_weight="balanced"
    )
    fm.fit(X_train, y_train, X_test, y_test)
    print(fm.score(X_test, y_test))
    print(fm.fm_nn.lr.weight.data)
    try:
        print(fm.fm_nn.embd.data)
    except:
        pass
