#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-12
# @Contact    : qichun.tang@bupt.edu.cn
import os
import tempfile
import warnings
from copy import copy
from uuid import uuid4

import numpy as np
import xlearn as xl
from sklearn.utils import check_X_y
from xlearn import create_fm, create_linear, create_ffm
from xlearn._sklearn import BaseXLearnModel

from xenon_ext.occ_model.base_occ import BaseOCC


def fit(self, X, y=None, fields=None,
        is_lock_free=True, is_instance_norm=True,
        eval_set=None, is_quiet=False):
    """ Fit the XLearn model given feature matrix X and label y

    :param X: array-like or a string specifying file location
              Feature matrix
    :param y: array-like
              Label
    :param fields: array-like
              Fields for FFMModel. Default as None
    :param is_lock_free: is using lock-free training
    :param is_instance_norm: is using instance-wise normalization
    :param eval_set: a 2-element list representing (X_val, y_val) or a string specifying file location
    :param is_quiet: is training model quietly
    """

    if self.model_type == 'fm':
        self._XLearnModel = create_fm()
    elif self.model_type == 'lr':
        self._XLearnModel = create_linear()
    elif self.model_type == 'ffm':
        self._XLearnModel = create_ffm()
    else:
        raise Exception('model_type must be fm, ffm or lr')
    # 核外计算OCC
    self._XLearnModel.setOnDisk()
    # create temporary files for training data
    temp_train_file = tempfile.NamedTemporaryFile(delete=True)

    if y is None:
        assert isinstance(X, str), 'X must be a string specifying training file location' \
                                   ' when only X specified'
        self._XLearnModel.setTrain(X)

    else:
        X, y = check_X_y(X, y, accept_sparse=['csr'], y_numeric=True, multi_output=False)

        if self.model_type == 'ffm':
            assert fields is not None, 'Must specify fields in FFMModel'
            self.fields = fields

        # convert data into libsvm/libffm format for training
        # TODO: replace conversion with DMatrix
        self._convert_data(X, y, temp_train_file.name, fields=self.fields)
        self._XLearnModel.setTrain(temp_train_file.name)

    # TODO: find out what task need to set sigmoid
    if self.task == 'binary':
        self._XLearnModel.setSigmoid()

    # set lock-free, is quiet training and instance-wise normalization
    if not is_lock_free:
        self._XLearnModel.disableLockFree()

    if is_quiet:
        self._XLearnModel.setQuiet()

    if not is_instance_norm:
        if self.model_type in ['fm', 'ffm']:
            self._XLearnModel.disableNorm()
        else:
            warnings.warn('Setting is_instance_norm to False is ignored. It only applies to fm or ffm.')

    params = self.get_xlearn_params()

    # check if validation set exists or not
    if eval_set is not None:
        if isinstance(eval_set, str):
            self._XLearnModel.setValidate(eval_set)
        else:
            if not (isinstance(eval_set, list) and len(eval_set) == 2):
                raise Exception('eval_set must be a 2-element list')

            # extract validation data
            X_val, y_val = check_X_y(eval_set[0], eval_set[1],
                                     accept_sparse=['csr'],
                                     y_numeric=True,
                                     multi_output=False)

            temp_val_file = tempfile.NamedTemporaryFile(delete=True)
            self._convert_data(X_val, y_val, temp_val_file.name, fields=self.fields)
            self._XLearnModel.setValidate(temp_val_file.name)

    # set up files for storing weights
    self._XLearnModel.setTXTModel(self._temp_weight_file.name)

    # fit model
    self._XLearnModel.fit(params, self._temp_model_file.name)

    # acquire weights
    self._parse_weight(self._temp_weight_file.name)

    # remove temporary files for training
    self._remove_temp_file(temp_train_file)


class XlearnEstimator(BaseOCC):
    def __init__(
            self,
            model_type="lr",
            block_size=50,
            lr=0.2,
            k=4,
            reg_lambda=0.0001,
            epoch=5,
            stop_window=2,
            opt='adagrad',  # adagrad, ftrl
            alpha=1,
            beta=1,
            lambda_1=1,
            lambda_2=1,
            is_instance_norm=True
    ):
        super(XlearnEstimator, self).__init__()
        self.is_instance_norm = is_instance_norm
        self.lambda_2 = lambda_2
        self.lambda_1 = lambda_1
        self.beta = beta
        self.alpha = alpha
        self.opt = opt
        self.stop_window = stop_window
        self.epoch = epoch
        self.reg_lambda = reg_lambda
        self.k = k
        self.lr = lr
        self.block_size = block_size
        self.model_type = model_type
        if self.is_classification:
            if self.target_type == "binary":
                self.task = "binary"
            elif self.target_type == "multiclass":
                raise NotImplementedError(f'xlearn 暂不支持多分类')
            else:
                raise ValueError(f"Invalid target_type {self.target_type}!")
        else:
            self.task = "reg"
        # fixme: 用于解决xlearn在xenon镜像中的问题
        # https://github.com/aksnzhy/xlearn/issues/215
        os.environ["USER"] = "test"

    def parse_model(self, model: BaseXLearnModel):
        _temp_model_file = model._temp_model_file.name
        with open(_temp_model_file, 'rb') as f:
            self.model_content = f.read()
        self.model_weights = self.model.weights

    # todo: 写成上下文感知的形式，如 with predict():
    def before_predict(self):
        self.model = self.get_model()
        if self.model_type == "lr":
            self.model._XLearnModel = create_linear()
        else:
            self.model._XLearnModel = create_fm()
        _temp_model_file = tempfile.NamedTemporaryFile(delete=True)
        _temp_model_file.name = f"/tmp/{uuid4().hex}"
        self.model._temp_model_file = _temp_model_file
        self.model.weights = self.model_weights
        with open(_temp_model_file.name, 'wb') as f:
            f.write(self.model_content)

    def after_predict(self):
        self.model = None

    def get_model(self):
        if self.model_type == "lr":
            klass = xl.LRModel
        elif self.model_type == "fm":
            klass = xl.FMModel
        elif self.model_type == "ffm":
            klass = xl.FFMModel
        else:
            raise NotImplementedError
        model = klass(
            task=self.task,
            block_size=self.block_size,
            lr=self.lr,
            k=self.k,
            reg_lambda=self.reg_lambda,
            epoch=self.epoch,
            stop_window=self.stop_window,
            opt=self.opt,
            alpha=self.alpha,
            beta=self.beta,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            verbose=0

        )
        return model

    def _occ_train(self, i):
        model = self.get_model()
        fit(
            model,
            X=self.train_path_list[i],
            eval_set=self.valid_path_list[i],
            is_quiet=True,
            # is_lock_free=False,
            is_instance_norm=self.is_instance_norm
        )
        ret = copy(self)
        ret.model = model
        ret.parse_model(model)
        return ret

    def _occ_validate(self, i, model):
        model.before_predict()
        xl_model: BaseXLearnModel = model.model
        # 确保： sigmoid
        label = self.labels[i]
        y_pred = xl_model.predict(self.valid_path_list[i])
        if self.task == 'binary':
            y_pred = y_pred[:, None]
            y_pred = np.hstack([1 - y_pred, y_pred])
        model.after_predict()
        return label, y_pred

    def predict(self, X):
        if self.is_classification:
            return self.predict_proba(X).argmax(axis=1)
        else:
            return self.predict_proba(X)

    def predict_proba(self, X):
        self.before_predict()
        xl_model: BaseXLearnModel = self.model
        y_pred = xl_model.predict(X)
        if self.task == 'binary':
            y_pred = y_pred[:, None]
            y_pred = np.hstack([1 - y_pred, y_pred])
        self.after_predict()
        return y_pred


class XlearnClassifier(XlearnEstimator):
    is_classification = True


class XlearnRegressor(XlearnEstimator):
    is_classification = False
