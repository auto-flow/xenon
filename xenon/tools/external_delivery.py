#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com

from mlxtend.classifier import StackingClassifier
from sklearn.pipeline import Pipeline
from xenon.core.base import XenonEstimator
from xenon.ensemble.mean.regressor import MeanRegressor as OldMeanRegressor
from xenon.ensemble.stack.classifier import StackClassifier as OldStackClassifier
from xenon.ensemble.stack.regressor import StackRegressor as OldStackRegressor
from xenon.ensemble.vote.classifier import VoteClassifier as OldVoteClassifier
from xenon.estimator.wrap_lightgbm import LGBMClassifier as OldLGBMClassifier
from xenon.estimator.wrap_lightgbm import LGBMEstimator as OldLGBMEstimator
from xenon.estimator.wrap_lightgbm import LGBMRegressor as OldLGBMRegressor

from xenon_ext.mean.regressor import MeanRegressor
from xenon_ext.stack.classifier import StackClassifier
from xenon_ext.stack.regressor import StackRegressor
from xenon_ext.vote.classifier import VoteClassifier
from xenon_ext.wrap_lightgbm import LGBMRegressor, LGBMClassifier


def update_klass(component, new_component):
    for k, v in component.__dict__.items():
        if hasattr(new_component, k):
            setattr(new_component, k, v)
    return new_component


def transform_pipeline(workflow):
    steps = []
    for name, wrap_component in workflow.steps:
        component = wrap_component.component
        update_mapper = {
            OldLGBMClassifier: LGBMClassifier,
            OldLGBMRegressor: LGBMRegressor,
        }
        if isinstance(component, OldLGBMEstimator):
            updated_klass = update_mapper[component.__class__]
            new_component = updated_klass()
            update_klass(component, new_component)
            component = new_component
        else:
            component = wrap_component.component
        steps.append([name, component])
    return Pipeline(steps)


def transform_bagging_model(component):
    update_mapper = {
        OldMeanRegressor: MeanRegressor,
        OldVoteClassifier: VoteClassifier,
    }
    updated_klass = update_mapper[component.__class__]
    new_component = updated_klass([])
    update_klass(component, new_component)
    new_component.models = [transform_pipeline(model) for model in new_component.models]
    return new_component


def transform_stacking_model(component):
    update_mapper = {
        OldStackClassifier: StackClassifier,
        OldStackRegressor: StackRegressor,
    }
    updated_klass = update_mapper[component.__class__]
    new_component = updated_klass()
    update_klass(component, new_component)
    new_component.estimators_list = [[transform_pipeline(estimator) for estimator in estimators] for estimators in
                                     new_component.estimators_list]
    return new_component


def transform_xenon(xenon: XenonEstimator):
    xenon_estimator = xenon.estimator
    if isinstance(xenon_estimator, (OldVoteClassifier, OldMeanRegressor)):
        return transform_bagging_model(xenon_estimator)
    elif isinstance(xenon_estimator, (OldStackClassifier, OldStackRegressor)):
        return transform_stacking_model(xenon_estimator)
    else:
        raise ValueError


if __name__ == '__main__':
    from joblib import load, dump
    import pandas as pd

    xenon1 = load("/home/tqc/PycharmProjects/Xenon/savedpath/ensemble_test_1/experiment_292_best_model.bz2")
    xenon2 = load("/home/tqc/PycharmProjects/Xenon/savedpath/clf_test_1/experiment_286_best_model.bz2")
    workflows = [estimators[0] for estimators in xenon1.estimator.estimators_list]
    workflow = workflows[0]
    pipeline = transform_pipeline(workflow)
    AP = pd.read_csv("/home/tqc/PycharmProjects/Xenon/data/job_33483_result/data/feature/AP2D.csv")
    data = pd.read_csv("/home/tqc/PycharmProjects/Xenon/data/job_33483_result/data/data.csv")
    df = AP.merge(data[["NAME", "active"]], on="NAME")
    df.pop('NAME')
    y = df.pop('active')
    X = df
    print(workflow[-1].component.__class__)
    print(pipeline[-1].__class__)
    pipelines = [(f"{i}", transform_pipeline(workflow)) for i, workflow in enumerate(workflows)]
    meta = xenon1.estimator.meta_learner
    stack = StackingClassifier(pipelines, meta)
    stack.final_estimator_ = meta
    stack.estimators_ = pipelines
    dump(pipeline, "pipeline.bz2")
    print(pipeline.score(X, y))
    trans1 = transform_xenon(xenon1)
    trans2 = transform_xenon(xenon2)
    print(trans1.score(X, y))
    print(trans2.score(X, y))
    dump(trans1, "trans1.bz2")
    dump(trans2, "trans2.bz2")

