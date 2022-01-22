#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-07
# @Contact    : qichun.tang@bupt.edu.cn
from warnings import filterwarnings

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import Lasso, LogisticRegression

filterwarnings('ignore')


# 通过面向对象的方法设计特征筛选
class DummyFeatureSelection():

    def select(self, X, y, columns, model_type):
        '通过这个函数筛选特征，最后返回【筛选后的特征矩阵、columns】'
        self._modeling_mask(X.values, y, model_type)
        columns = pd.Series(columns)[self.mask].tolist()
        return X.loc[:, self.mask], columns

    def _modeling_mask(self, X, y, model_type):
        self.coef = np.zeros([X.shape[1]], dtype=float)
        self.mask = np.ones([X.shape[1]], dtype=bool)

    def dump_model(self, path):
        pass


class ModelBased_FeatureSelection(DummyFeatureSelection):
    def __init__(self, params):
        self.params = params

    def dump_model(self, path):
        dump(self.model, path)


class L1Linear_FeatureSelection(ModelBased_FeatureSelection):
    def _modeling_mask(self, X, y, model_type):
        '''_modeling_mask'''
        '''
        Note:
        分类器 LogisticRegression 的 参数 C 是 1/正则化系数
        回归器 Lasso 的 参数 alpha 是 正则化系数
        调参经验：
        - C越大，惩罚力度越小，保留特征越多
        - 样本越少，数据越稀疏，相同的正则化参数下保留的特征越少
        两个例子：
        - 对于20w~40w样本，9000特征的数据，C设为0.5
        - 对于1000以下样本，9000特征的数据，C设为 10~100
        - 具体设多少可以多跑几个preprocess，然后看特征保留率
        - todo: 更智能的做法
        这行代码将分类与回归的正则化系数统一起来用C表示 ↓
        '''
        if model_type == "reg" and "C" in self.params:
            self.params['alpha'] = 1 / self.params.pop("C")
        ''' 这行代码将分类与回归的正则化系数统一起来用C表示 ↑'''
        if model_type == 'reg':
            params = dict(random_state=0)
            params.update(self.params)
            model = Lasso(**params)
        else:
            params = dict(penalty='l1', random_state=0, solver='liblinear')
            params.update(self.params)
            model = LogisticRegression(**params)
        model.fit(X, y)
        coef = model.coef_
        if coef.ndim == 2:
            coef = coef.sum(axis=0)
        self.model = model
        self.coef = coef
        self.mask = coef != 0


class GBDT_FeatureSelection(ModelBased_FeatureSelection):
    def _modeling_mask(self, X, y, model_type):
        params = dict(boosting_type='gbdt', random_state=0)
        params.update(self.params)
        if model_type == 'clf':
            model = LGBMClassifier(**self.params)
        else:
            model = LGBMRegressor(**self.params)
        model.fit(X, y)
        coef = model.feature_importances_
        self.model = model
        self.coef = coef
        self.mask = coef != 0


class RandomForest_FeatureSelection(ModelBased_FeatureSelection):
    def _modeling_mask(self, X, y, model_type):
        params = dict(boosting_type='rf', random_state=0, bagging_freq=1,
                      bagging_fraction=0.8)
        params.update(self.params)
        if model_type == 'clf':
            model = LGBMClassifier(**params)
        else:
            model = LGBMRegressor(**params)
        model.fit(X, y)
        coef = model.feature_importances_
        self.model = model
        self.coef = coef
        self.mask = coef != 0


if __name__ == '__main__':
    from sklearn.datasets import load_boston, load_digits

    for cls in [RandomForest_FeatureSelection, GBDT_FeatureSelection, DummyFeatureSelection,
                L1Linear_FeatureSelection]:
        print(cls)
        X, y = load_digits(n_class=2, return_X_y=True)
        X1, columns = cls({}).select(X, y, list(range(64)), 'clf')
        print(X1.shape, len(columns))

        X, y = load_digits(n_class=10, return_X_y=True)
        X1, columns = cls({}).select(X, y, list(range(64)), 'clf')
        print(X1.shape, len(columns))

        X, y = load_boston(True)
        X1, columns = cls({}).select(X, y, list(range(13)), 'reg')
        print(X1.shape, len(columns))
        print('=' * 100)
        print()
