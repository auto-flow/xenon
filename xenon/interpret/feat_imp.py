#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd

from xenon.core.base import XenonEstimator
from xenon.ensemble.mean.regressor import MeanRegressor
from xenon.ensemble.stack.base import StackEstimator
from xenon.ensemble.vote.classifier import VoteClassifier
from xenon.utils.logging_ import get_logger
from xenon.workflow.ml_workflow import ML_Workflow

logger = get_logger(__name__)


def get_feature_importances_in_workflow(ml_workflow: ML_Workflow, columns):
    selected_columns = deepcopy(columns)
    df = pd.DataFrame(index=columns)
    for i, (name, wrap_component) in enumerate(ml_workflow.steps[:-1]):
        component = wrap_component.component
        if hasattr(component, "get_support"):
            mask = component.get_support()
            feature_importances_ = component.estimator_.feature_importances_
            df.loc[selected_columns, name] = feature_importances_
            selected_columns = selected_columns[mask]
    estimator = ml_workflow[-1].component
    name = ml_workflow.steps[-1][0]
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
    return df


def get_feature_importances_in_workflows(ml_workflows: List[ML_Workflow], columns):
    dfs = [get_feature_importances_in_workflow(ml_workflow, columns) for ml_workflow in ml_workflows]
    values = np.array([df.values for df in dfs])
    df = pd.DataFrame(values.mean(axis=0), index=dfs[0].index, columns=dfs[0].columns)
    return df


def get_feature_importances_in_xenon(xenon: XenonEstimator) -> List[pd.DataFrame]:
    columns = xenon.data_manager.columns
    xenon_estimator = xenon.estimator
    if isinstance(xenon_estimator, (MeanRegressor, VoteClassifier)):
        return [get_feature_importances_in_workflow(xenon_estimator.models, columns)]
    elif isinstance(xenon_estimator, StackEstimator):
        return [get_feature_importances_in_workflows(estimators, columns) for estimators in
                xenon_estimator.estimators_list]


if __name__ == '__main__':
    from joblib import load

    xenon = load("/home/tqc/PycharmProjects/Xenon/savedpath/ensemble_test_1/experiment_292_best_model.bz2")
    columns = xenon.data_manager.columns
    estimators = xenon.estimator.estimators_list[0]
    results = get_feature_importances_in_xenon(xenon)
    print(results)
