#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-06-05
# @Contact    : qichun.tang@bupt.edu.cn
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston, load_digits


def build_iris():
    X, y = load_iris(True)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('iris.csv', index=False)


def build_binary_iris():
    X, y = load_iris(True)
    y[np.isin(y, [0, 1])] = 0
    y[np.isin(y, [2, 3])] = 1
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('binary_iris.csv', index=False)


def build_binary_digits():
    X, y = load_digits(2, True)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('binary_digits.csv', index=False)


def build_digits():
    X, y = load_digits(10, True)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('digits.csv', index=False)


def build_middle_digits():
    X, y = load_digits(5, True)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('middle_digits.csv', index=False)


def build_small_digits():
    X, y = load_digits(3, True)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('small_digits.csv', index=False)


def build_boston():
    X, y = load_boston(True)
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv('small_digits.csv', index=False)


if __name__ == '__main__':
    build_iris()
    build_binary_iris()
    build_digits()
    build_binary_digits()
    build_small_digits()
    build_middle_digits()
    build_digits()
