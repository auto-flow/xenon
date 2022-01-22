#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd

from autoflow.core.base import AutoFlowEstimator
from autoflow.ensemble.mean.regressor import MeanRegressor
from autoflow.ensemble.stack.base import StackEstimator
from autoflow.ensemble.vote.classifier import VoteClassifier
from autoflow.utils.logging_ import get_logger
from autoflow.workflow.ml_workflow import ML_Workflow

logger = get_logger(__name__)


def get_feature_importances_in_workflow(ml_workflow: ML_Workflow, columns) -> Tuple[pd.DataFrame, pd.Index]:
    selected_columns = deepcopy(columns)
    df = pd.DataFrame(index=columns)
    # 处理特征工程部分
    for i, (name, wrap_component) in enumerate(ml_workflow.steps[:-1]):
        component = wrap_component.component
        if hasattr(component, "get_support"):
            # 这里为什么try-except ？
            # 因为autoflow_ext.feature_selection.flexible_feature_selector.FlexibleFeatureSelector#get_support
            # 可能raise Exception
            try:
                mask = component.get_support()
                # 这里为什么try-except ？
                # 因为 FlexibleFeatureSelector 和 自研的 VarianceThreshold 只有 feature_importances_
                # 没有 estimator_.feature_importances_
                try:
                    feature_importances_ = component.estimator_.feature_importances_
                except:
                    feature_importances_ = component.feature_importances_
                df.loc[selected_columns, name] = feature_importances_
                selected_columns = selected_columns[mask]
            except:
                pass
    # 处理模型部分
    estimator = ml_workflow[-1].component
    name = ml_workflow.steps[-1][0]
    # 先判断lightgbm（重叠条件优先判断）
    if hasattr(estimator, "model") and hasattr(estimator.model, "feature_importance"):
        feat_imp = estimator.model.feature_importance('gain')
        feat_imp /= feat_imp.sum()
    elif hasattr(estimator, "feature_importances_"):
        feat_imp = estimator.feature_importances_
    else:
        feat_imp = None
    if feat_imp is not None:
        df.loc[selected_columns, name] = feat_imp
    df[pd.isna(df)] = 0
    df.index.name = "columns"
    return df, selected_columns


def get_feature_importances_in_workflows(ml_workflows: List[ML_Workflow], columns) -> Tuple[pd.DataFrame, pd.Index]:
    # 获取所有的kfold模型的特征重要度
    outputs = [get_feature_importances_in_workflow(ml_workflow, columns) for ml_workflow in ml_workflows]
    values = np.array([output[0].values for output in outputs])
    # 对筛出来的列名做投票
    k_selected_columns_list = [output[1] for output in outputs]
    k = len(k_selected_columns_list)     # 比如=5
    m = k // 2 + 1 if k % 2 else k // 2  # 比如=3
    res_selected_columns = []
    for column in columns:
        # 开始投票
        hit_nums = 0
        for selected_columns in k_selected_columns_list:
            if column in selected_columns:
                hit_nums += 1
        if hit_nums >= m:
            res_selected_columns.append(column)
    # 对特征重要度求平均
    df = pd.DataFrame(values.mean(axis=0), index=outputs[0][0].index, columns=outputs[0][0].columns)
    return df, pd.Index(res_selected_columns)


def get_feature_importances_in_autoflow(autoflow: AutoFlowEstimator) -> List[Tuple[pd.DataFrame, pd.Index]]:
    columns = autoflow.data_manager.columns
    autoflow_estimator = autoflow.estimator
    if isinstance(autoflow_estimator, (MeanRegressor, VoteClassifier)):
        return [get_feature_importances_in_workflows(autoflow_estimator.models, columns)]
    elif isinstance(autoflow_estimator, StackEstimator):
        return [get_feature_importances_in_workflows(estimators, columns) for estimators in
                autoflow_estimator.estimators_list]


if __name__ == '__main__':
    from joblib import load

    autoflow = load("/home/tqc/Project/AutoFlow/savedpath/test_feat_imp_clf_single/experiment_674_best_model.bz2")
    columns = autoflow.data_manager.columns
    results = get_feature_importances_in_autoflow(autoflow)
    print(results)
