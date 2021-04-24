#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-20
# @Contact    : qichun.tang@bupt.edu.cn
from functools import partial
from logging import getLogger
from math import ceil
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from torch import nn
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits

from xenon_ext.utils import parse_n_jobs
from .ftrl import FTRL

logger = getLogger(__name__)


class FMLayer(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FMLayer, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class FMModel(nn.Module):
    def __init__(self, n_inputs, n_outputs=1, k=2, use_fm=True):
        super(FMModel, self).__init__()
        self.use_fm = use_fm
        self.k = k
        self.lr = nn.Linear(n_inputs, n_outputs)  # 包含了b
        self.initializing_modules(self.lr)
        if self.use_fm:
            self.embd = nn.Parameter(torch.randn([1, n_inputs, k]))
            self.fm = FMLayer()

    def forward(self, X):
        lr_out = self.lr(X)
        if self.use_fm:
            embedding = X[:, :, None].repeat_interleave(self.k, dim=2) * self.embd
            fm_out = self.fm(embedding)
            out = fm_out + lr_out
        else:
            out = lr_out
        out = out.flatten()
        return out

    def initializing_modules(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            m.bias.data.zero_()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FMBaseEstimator(BaseEstimator):
    is_classification = None

    def __init__(
            self,
            # 功能性参数
            n_jobs=-1, random_state=1000, class_weight=None,
            batch_size=1024, max_epoch=200, early_stop_rounds=10,
            tol=1e-3,
            device="auto",
            # 优化器参数
            optimizer="sgd", lr=0.1, weight_decay=1e4,
            alpha=1, beta=1, l1=1, l2=1,
            # 模型参数
            k=2, use_fm=True
    ):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.l1 = l1
        self.alpha = alpha
        self.beta = beta
        self.l2 = l2
        self.tol = tol
        self.early_stop_rounds = early_stop_rounds
        self.use_fm = use_fm
        self.weight_decay = weight_decay
        self.k = k
        self.class_weight = class_weight
        self.n_jobs = parse_n_jobs(n_jobs)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_epoch = max_epoch
        self.lr = lr
        self.rng = check_random_state(random_state)
        self.device = device

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_valid: Optional[np.ndarray] = None,
            y_valid: Optional[np.ndarray] = None,
            sample_weight=None
    ):
        torch.manual_seed(self.rng.randint(0, 10000))
        torch.set_num_threads(self.n_jobs)
        fm_nn: nn.Module = FMModel(X.shape[1], 1, k=self.k, use_fm=self.use_fm)
        fm_nn.to(self.device)
        # todo: FTRL
        if self.optimizer == "adam":
            nn_optimizer = torch.optim.Adam(fm_nn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            nn_optimizer = torch.optim.SGD(fm_nn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "ftrl":
            nn_optimizer = FTRL(
                fm_nn.parameters(),
                alpha=self.alpha,
                beta=self.beta,
                l1=self.l1,
                l2=self.l2,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        start_time = time()
        # 处理数据
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X_valid, pd.DataFrame):
            X_valid = X_valid.values
        X = torch.from_numpy(X.astype('float32')).to(self.device)
        self.y_scaler = None
        if not self.is_classification:
            self.y_scaler = StandardScaler().fit(y[:, np.newaxis])
            y = self.y_scaler.transform(y[:, np.newaxis]).flatten()
            if y_valid is not None:
                y_valid = self.y_scaler.transform(y_valid[:, np.newaxis]).flatten()

        y = torch.from_numpy(y).double().to(self.device)
        if X_valid is not None and y_valid is not None:
            X_valid = torch.from_numpy(X_valid.astype('float32')).to(self.device)
            y_valid = torch.from_numpy(y_valid).double().to(self.device)
        if self.is_classification and self.class_weight == "balanced":
            unique, counts = np.unique(y, return_counts=True)
            assert np.all(unique == np.array([0, 1]))
            weight = counts[0] / counts[1]
            weight = torch.tensor(weight, dtype=torch.float32)
        else:
            weight = None
        # 早停窗口
        early_stopping_window = np.zeros([self.early_stop_rounds]) + np.inf
        # 损失函数
        if self.is_classification:
            loss_func = partial(binary_cross_entropy_with_logits,
                                pos_weight=weight)
        else:
            loss_func = mse_loss
        # 进行迭代
        for epoch_index in range(0, self.max_epoch):
            fm_nn.train(True)
            # batch
            permutation = self.rng.permutation(len(y))
            # 构造batch索引
            batch_ixs = []
            for i in range(ceil(len(y) / self.batch_size)):
                start = min(i * self.batch_size, len(y))
                end = min((i + 1) * self.batch_size, len(y))
                batch_ix = permutation[start:end]
                if end - start < self.batch_size:
                    diff = self.batch_size - (end - start)
                    diff = min(diff, start)
                    batch_ix = np.hstack([batch_ix, self.rng.choice(permutation[:start], diff, replace=False)])
                batch_ixs.append(batch_ix)
            # batch 迭代
            train_losses = []
            for batch_ix in batch_ixs:
                nn_optimizer.zero_grad()
                outputs = fm_nn(X[batch_ix, :])
                loss = loss_func(outputs.double(), y[batch_ix])
                loss.backward()
                train_losses.append(loss.detach().cpu().numpy())

                nn_optimizer.step()
            train_loss = np.mean(train_losses)
            msg = (f"epoch: {epoch_index:03d}, train_loss: {train_loss:.4f}")
            # 早停判断
            if X_valid is not None and y_valid is not None:
                fm_nn.eval()
                outputs = fm_nn(X_valid)
                valid_loss = loss_func(outputs.double(), y_valid).detach().cpu().numpy()
                msg += f", valid_loss: {valid_loss:.4f}"
                if valid_loss >= early_stopping_window.max() - self.tol:
                    print(f"early stop")
                    break
                early_stopping_window[epoch_index % self.early_stop_rounds] = valid_loss
            # 信息打印
            if epoch_index % 100 == 0:
                print(msg)
        end = time()
        print(f"{fm_nn.__class__.__name__} training time = {end - start_time:.2f}s")
        fm_nn.eval()
        self.fm_nn = fm_nn
        return self

    def _predict(self, X):
        return self.fm_nn(
            torch.from_numpy(X.astype('float32')).to(self.device)
        ).detach().cpu().numpy()


class FMClassifier(FMBaseEstimator, ClassifierMixin):
    '''当前分类器只能处理二分类，因为FM层输出的是一维向量，没法弄成softmax'''
    is_classification = True

    def predict_proba(self, X):
        proba_1 = sigmoid(self._predict(X))[:, np.newaxis]
        proba_0 = 1 - proba_1
        return np.hstack([proba_0, proba_1])

    def predict(self, X):
        return (self._predict(X) > 0).astype(int)


class FMRegressor(FMBaseEstimator, RegressorMixin):
    is_classification = False

    def predict(self, X):
        y_pred = self._predict(X)
        return self.y_scaler.inverse_transform(y_pred[:, np.newaxis]).flatten()
