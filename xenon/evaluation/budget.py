#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import json5 as json
import numpy as np
from joblib import Memory
from collections import Counter

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import type_of_target

from xenon.data_container import DataFrameContainer, NdArrayContainer


def get_stratified_sampling_index(array: np.ndarray, proportion: float, random_state: int):
    array = array.astype("float32")
    if type_of_target(array) == "continuous":
        if len(array) < 10:
            n_bins = 2
        else:
            n_bins = 5
        kbins = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
        array = kbins.fit_transform(array.reshape(-1, 1)).squeeze()
    assert len(array.shape) == 1
    rng = np.random.RandomState(random_state)
    labels = Counter(array)
    origin_index = np.arange(len(array), dtype='int32')
    results = []
    for label in labels:
        mask = (array == label)
        masked_index = origin_index[mask]
        L = max(1, round(len(masked_index) * proportion))
        samples = rng.choice(masked_index, L, replace=False)
        results.append(samples)
    result = np.hstack(results)
    rng.shuffle(result)
    return result


def implement_subsample_budget(
        X_train: DataFrameContainer, y_train: NdArrayContainer,
        budget, random_state: int
) -> Tuple[DataFrameContainer, NdArrayContainer]:
    sub_sample_index = get_stratified_sampling_index(y_train.data, budget, random_state)
    # sub sampling X_train, y_train
    X_train = X_train.sub_sample(sub_sample_index)
    y_train = y_train.sub_sample(sub_sample_index)
    return X_train, y_train
