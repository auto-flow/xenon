#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-08
# @Contact    : qichun.tang@bupt.edu.cn
import datetime
import os
from collections import defaultdict
from time import time

import numpy as np
from joblib import load

from neutron.hdl import layering_config
from xenon import ResourceManager
from xenon.metrics import calculate_score, calculate_confusion_matrix
from xenon.utils.hash import get_hash_of_dict
from xenon.utils.logging_ import get_logger
from xenon.utils.ml_task import MLTask
from xenon4bigdata.est_mapper import mapper
from xenon_ext.occ_model.base_occ import BaseOCC


class Evaluator():
    def __init__(self, ml_task: MLTask, metric, resource_manager: ResourceManager, estimator_params: dict):
        self.estimator_params = estimator_params
        self.resource_manager = resource_manager
        self.metric = metric
        self.ml_task = ml_task
        self.logger = get_logger(self)
        self.y_true_indexes = load(os.environ["DATAPATH"] + "/valid_set_indexes.pkl")

    def __call__(self, config: dict, budget=1):
        config_id = get_hash_of_dict(config)
        start_time = datetime.datetime.now()
        start = time()
        config_dict = layering_config(config)
        sub_dict: dict = config_dict["estimator"]
        # 算法选择，超参优化
        algo_select, hyper_params = sub_dict.popitem()
        # 根据estimator_params设置固定参数
        fixed_params = self.estimator_params.get(algo_select, {})
        for k, v in fixed_params.items():
            hyper_params[k] = v
        klass = mapper[algo_select][self.ml_task.mainTask]
        occ_model: BaseOCC = klass(**hyper_params)
        models = occ_model.occ_train()
        labels, predictions = occ_model.occ_validate(models)
        losses = []
        all_scores = []
        additional_info = {}
        confusion_matrices = []
        y_preds = []
        for i, (label, prediction) in enumerate(zip(labels, predictions)):
            loss, all_score = self.loss(label, prediction)
            y_preds.append(prediction)
            losses.append(loss)
            all_scores.append(all_score)
            if self.ml_task.mainTask == "classification":
                confusion_matrices.append(calculate_confusion_matrix(label, prediction))
        if self.ml_task.mainTask == "classification":
            additional_info["confusion_matrices"] = confusion_matrices
        if len(losses) > 0:
            final_loss = float(np.array(losses).mean())
        else:
            final_loss = 65535
        if len(all_scores) > 0 and all_scores[0]:
            all_score = defaultdict(list)
            for cur_all_score in all_scores:
                if isinstance(cur_all_score, dict):
                    for key, value in cur_all_score.items():
                        all_score[key].append(value)
                else:
                    self.logger.warning(f"TypeError: cur_all_score is not dict.\ncur_all_score = {cur_all_score}")
            for key in all_score.keys():
                all_score[key] = float(np.mean(all_score[key]))
        else:
            all_score = {}
            all_scores = []
        end_time = datetime.datetime.now()

        info = {
            "warning_info": "",
            "loss": final_loss,
            "losses": losses,
            "all_score": all_score,
            "all_scores": all_scores,
            "models": models,
            "finally_fit_model": None,
            "y_true_indexes": self.y_true_indexes,
            "y_preds": y_preds,
            "intermediate_results": [],
            "status": "SUCCESS",
            "failed_info": "",
            "start_time": start_time,
            "end_time": end_time,
            "additional_info": additional_info
        }
        cost_time = time() - start
        info["config_id"] = config_id
        info["instance_id"] = ""
        info["run_id"] = ""
        # 相当于未处理前的config
        info["dict_hyper_param"] = config
        estimator = algo_select
        info["estimator"] = estimator
        info["cost_time"] = cost_time
        self.resource_manager.insert_trial_record(info)
        return info["loss"]

    def loss(self, y_true, y_hat):
        score, true_score = calculate_score(
            y_true, y_hat, self.ml_task.mainTask, self.metric,
            should_calc_all_metric=True)
        if isinstance(score, dict):
            err = self.metric._optimum - score[self.metric.name]
            all_score = true_score
        elif isinstance(score, (int, float)):
            err = self.metric._optimum - score
            all_score = None
        else:
            raise TypeError

        return err, all_score
