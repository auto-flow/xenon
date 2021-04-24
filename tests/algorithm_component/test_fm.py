#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-24
# @Contact    : qichun.tang@bupt.edu.cn
import os

from sklearn import datasets
from sklearn.model_selection import train_test_split

from xenon.data_container import DataFrameContainer
from xenon.data_container import NdArrayContainer
from xenon.tests.base import LocalResourceTestCase
from xenon.workflow.components.classification.pytorch_fm import FMClassifier
from xenon.workflow.components.regression.pytorch_fm import FMRegressor


class TestFM(LocalResourceTestCase):
    def setUp(self) -> None:
        super(TestFM, self).setUp()
        self.plot_dir = os.getcwd() + "/test_iter_algorithm"
        from pathlib import Path
        Path(self.plot_dir).mkdir(parents=True, exist_ok=True)

    def test_classifier(self):
        X, y = datasets.load_digits(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)
        FMClassifier().fit(X_train, y_train, X_test, y_test)

    def test_regressor(self):
        X, y = datasets.load_boston(return_X_y=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                            random_state=0)
        X_train = DataFrameContainer("TrainSet", dataset_instance=X_train, resource_manager=self.mock_resource_manager)
        X_test = DataFrameContainer("TestSet", dataset_instance=X_test, resource_manager=self.mock_resource_manager)
        y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train, resource_manager=self.mock_resource_manager)
        y_test = NdArrayContainer("TestLabel", dataset_instance=y_test, resource_manager=self.mock_resource_manager)
        FMRegressor().fit(X_train, y_train, X_test, y_test)
