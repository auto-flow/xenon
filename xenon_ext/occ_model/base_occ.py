#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-08
# @Contact    : qichun.tang@bupt.edu.cn
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target


class BaseOCC(BaseEstimator):
    '''继承后，predict predict_prob也需要实现，因为这个类需要被加载去预测numpy.array数据'''
    is_classification = None

    def __init__(self):
        self.datapath = os.getenv("DATAPATH", ValueError("调用BaseOCC的时候需要指定DATAPATH环境变量"))
        i = 0
        train_path_list = []
        valid_path_list = []
        while True:
            train_path = self.datapath + f"/train_{i}.txt"
            valid_path = self.datapath + f"/valid_{i}.txt"
            if not (os.path.exists(train_path) and os.path.exists(valid_path)):
                break
            i += 1
            train_path_list.append(train_path)
            valid_path_list.append(valid_path)
        self.kfolds = i
        self.train_path_list = train_path_list
        self.valid_path_list = valid_path_list
        assert self.kfolds
        y = pd.read_csv(self.datapath + "/LABEL.csv")["LABEL"].values
        valid_set_indexes = load(self.datapath + "/valid_set_indexes.pkl")
        self.labels = [
            y[index]
            for index in valid_set_indexes
        ]
        self.target_type = type_of_target(y)
        self.n_classes = np.max(y) + 1

    def fit(self, X=None, y=None):
        return self

    def occ_train(self) -> List[BaseEstimator]:
        '''核外训练，返回kfolds个训练好的模型'''
        models = []
        for i in range(self.kfolds):
            model = self._occ_train(i)
            models.append(model)
        return models

    def _occ_train(self, i) -> BaseEstimator:
        raise NotImplementedError

    def occ_validate(self, models) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        '''返回一个二元组，分别是labels和验证集的predictions'''
        # 其长度大小为交叉验证(kfolds)的长度
        labels = []
        predictions = []
        for i in range(self.kfolds):
            label, prediction = self._occ_validate(i, models[i])
            labels.append(label)
            predictions.append(prediction)
        return labels, predictions

    def _occ_validate(self, i, model) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
