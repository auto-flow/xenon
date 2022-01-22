#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-09
# @Contact    : qichun.tang@bupt.edu.cn
'''
用于把【feature_select】产生的特征筛选器和【search】产生的学习器组装成一个【sklearn.pipeline.Pipeline】
注意：所有模块都放置在xenon_ext中，目的就是直接产出一个【对外交付】的模型，即依赖少量xenon代码（不含贝叶斯优化等核心代码）就
可以运行的模型
'''
import os
from pathlib import Path
from typing import List, Tuple

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

from xenon.utils.ml_task import MLTask
from xenon_ext.mean.regressor import MeanRegressor
from xenon_ext.selector import Xenon4bigdata_FeatureSelector
from xenon_ext.vote.classifier import VoteClassifier


def build_final_model_unused(models, ml_task: MLTask, columns_txt=None) -> Pipeline:
    '''之前的做法，已废弃'''
    if columns_txt is None:
        columns_txt = Path(os.environ['DATAPATH'] + "/columns.txt").read_text()
    return Pipeline([
        ('feature_selector',
         Xenon4bigdata_FeatureSelector(columns_txt.split(","))),
        ('estimator', VoteClassifier(models) if ml_task.mainTask == "classification" else MeanRegressor(models))
    ])


def build_final_model(transformers: List[Tuple[str, TransformerMixin]], models, ml_task: MLTask) -> Pipeline:
    '''
    当前的做法
    - preprocessing阶段 构造和保存Transformers
    - search阶段 被上传到nitrogen和数据库（也会用于拼接Pipeline）
    - ensemble阶段  的时候会下载Transformers用于拼接Pipeline
    '''
    return Pipeline(transformers + [
        ('estimator', VoteClassifier(models) \
            if ml_task.mainTask == "classification" else MeanRegressor(models))
    ])
