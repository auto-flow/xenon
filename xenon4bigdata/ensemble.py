#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-12
# @Contact    : qichun.tang@bupt.edu.cn

from xenon.ensemble.stack import classifier as inner_classifier, regressor as inner_regressor
from xenon4bigdata.pipeline import build_final_model
from xenon_ext.stack import classifier, regressor


def get_ensemble_model_by_trial_ids(resource_manager, task_id, trial_ids, metric):
    '''
    目前有两种做集成学习（stacking）的方法：
    1. 根据当前task_id，选取loss最小的k个trial_id
    2. 用户指定trial_ids，并且这些trial_ids可以task_id不同（但是训练数据需要是一致的）
    search阶段模型产生时采用方案1，fetch当前task_id下最好的trial_id对应的模型
    '''
    ml_task, y_true = resource_manager.get_ensemble_needed_info(task_id)  # y_true 是NdArrayContainer类型
    if metric is None:
        if ml_task.mainTask == "classification":
            metric = "mcc"
        else:
            metric = "r2"
    if not isinstance(metric, str):
        metric = metric.name

    estimator_list, y_true_indexes_list, y_preds_list, performance_list, scores_list, experiment_id_list = \
        resource_manager.load_estimators_in_trials(trial_ids, ml_task, metric)
    # 遍历experiment_id_list，通过get experiment得到 additional_info ->> transformers_dataset_id，下载对应的transformers对象
    transformers_list = []
    for experiment_id in experiment_id_list:
        records = resource_manager._get_experiment_record(experiment_id)
        assert len(records) > 0, ValueError(f"get {experiment_id} 异常，记录不存在?!")
        transformers_dataset_id = records[0]['additional_info'].get(
            'transformers_dataset_id',
            ValueError(f"{experiment_id} 异常，不含有 additional_info ->> transformers_dataset_id 字段"))
        cur_transformers = resource_manager.file_system.load_pickle(transformers_dataset_id)
        transformers_list.append(cur_transformers)
    models = []
    for transformers, estimator in zip(transformers_list, estimator_list):
        final_model = build_final_model(transformers, estimator, ml_task)
        models.append(final_model)
    # 只集成一个模型
    if len(models) == 1:
        return models[0]
    # 开始stacking
    if ml_task.mainTask == "classification":
        inner_klass = inner_classifier.StackClassifier
        klass = classifier.StackClassifier
    else:
        inner_klass = inner_regressor.StackRegressor
        klass = regressor.StackRegressor
    inner_stacker = inner_klass()
    stacker = klass()
    inner_stacker.fit_trained_data(models, y_preds_list, y_true_indexes_list, y_true.data)
    stacker.meta_learner = inner_stacker.meta_learner
    stacker.use_features_in_secondary = inner_stacker.use_features_in_secondary
    stacker.estimators_list = inner_stacker.estimators_list
    stacker.prediction_list = inner_stacker.prediction_list
    return stacker
    # if ml_task.mainTask == "classification":
    #     meta_cls = LogisticRegression
    #     meta_hps = {
    #         "penalty": "l2",
    #         "solver": "lbfgs",
    #         "C": {"_type": "loguniform", "_value": [0.01, 10], "_default": 0.1},
    #         # "fit_intercept": {"_type": "choice", "_value": [True, False], "_default": True},
    #         "random_state": 0,
    #         "max_iter": 100
    #     }
    # else:
    #     meta_cls = Ridge
    #     meta_hps = dict(
    #         alpha={"_type": "loguniform", "_value": [0.01, 10], "_default": 0.1},
    #         fit_intercept={"_type": "choice", "_value": [True, False], "_default": True},
    #         normalize=True,
    #         positive=True,  # force all coef_ to true
    #         random_state=42,
    #     )
