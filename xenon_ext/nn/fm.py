#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-20
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.utils import check_random_state
from torch import nn
import torch
from frozendict import frozendict
from math import ceil
from time import time
from typing import Optional, Callable

import numpy as np
import torch
from frozendict import frozendict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import type_of_target
from torch import nn
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from logging import getLogger
from sklearn.base import BaseEstimator,ClassifierMixin,RegressorMixin
from xenon_ext.utils import parse_n_jobs

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
    def __init__(self, n_inputs, n_outputs=1, k=2):
        super(FMModel, self).__init__()
        self.k = k
        self.lr = nn.Linear(n_inputs, n_outputs)  # 包含了b
        self.embd = nn.Parameter(torch.randn([1, n_inputs, k]))
        self.fm = FMLayer()

    def forward(self, X):
        embedding = X[:, :, None].repeat_interleave(self.k, dim=2) * self.embd
        fm_out = self.fm(embedding)
        lr_out = self.lr(X)
        return fm_out + lr_out


class FMBaseEstimator(BaseEstimator):
    def __init__(
            self,
            lr=1e-2, max_epoch=25,is_classification=True,
            random_state=1000, batch_size=1024, optimizer="adam", n_jobs=-1,
            class_weight=None, k=2
    ):
        self.is_classification = is_classification
        self.k = k
        self.class_weight = class_weight
        self.n_jobs = parse_n_jobs(n_jobs)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.random_state = random_state
        self.max_epoch = max_epoch
        self.lr = lr
        self.rng = check_random_state(random_state)

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            X_valid: Optional[np.ndarray] = None,
            y_valid: Optional[np.ndarray] = None,
    ):
        torch.manual_seed(self.rng.randint(0, 10000))
        torch.set_num_threads(self.n_jobs)
        if self.n_class is None:
            if type_of_target(y.astype("float")) == "continuous":
                self.n_class = 1
            else:
                self.n_class = np.unique(y).size

        fm_nn: nn.Module = FMModel(X.shape[1], 1, k=self.k)
        if self.optimizer == "adam":
            nn_optimizer = torch.optim.Adam(fm_nn.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            nn_optimizer = torch.optim.SGD(fm_nn.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer {self.optimizer}")
        start_time = time()
        y_tensor = torch.from_numpy(y).double()
        if self.n_class >= 2 and self.class_weight == "balanced":
            unique, counts = np.unique(y, return_counts=True)
            counts = counts / np.min(counts)
            weight = torch.from_numpy(1 / counts).double()
        else:
            weight = None
        for epoch_index in range(0, self.max_epoch):
            fm_nn.train(True)
            # batch
            permutation = self.rng.permutation(len(y))
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
            for batch_ix in batch_ixs:
                nn_optimizer.zero_grad()
                outputs = fm_nn()
                if self.n_class >= 2:
                    loss = binary_cross_entropy_with_logits(
                        outputs.double(), y_tensor[batch_ix], weight=weight)
                elif self.n_class == 1:
                    loss = mse_loss(outputs.flatten().double(), y_tensor[batch_ix])
                else:
                    raise ValueError
                loss.backward()
                nn_optimizer.step()
            fm_nn.max_epoch = epoch_index + 1
            # 早停判断
            if X_valid is not None and y_valid is not None:
                fm_nn.eval()
                # todo
        end = time()
        logger.info(f"{fm_nn.__class__.__name__} training time = {end - start_time:.2f}s")
        fm_nn.eval()
        return fm_nn

    def predict(self):
        pass


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    X, y = load_iris(True)
    X = MinMaxScaler().fit_transform(X)
    y[y != 0] = 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    fm = FMModel(4)
    fm(torch.from_numpy(X_train.astype('float32')))
