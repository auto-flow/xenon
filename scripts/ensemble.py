#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
###################################
# Insert current path to sys path #
###################################
import os
import sys

sys.path.insert(0, os.getcwd())
###################################
import logging
import os

from xenon import XenonClassifier, XenonRegressor
from scripts.utils import EnvUtils, save_current_expriment_model, save_info_json, process_previous_result_dataset, \
    print_xenon_path, display
from xenon.utils.ml_task import MLTask
from xenon.utils.logging_ import setup_logger
from xenon.ensemble.stack.base import StackEstimator
from xenon.resource_manager.http import HttpResourceManager

savedpath = os.getenv("SAVEDPATH", ".")
setup_logger(
    f"{savedpath}/xenon.log"
)
process_previous_result_dataset()
env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/ensemble.json")
env_utils.update()
logger = logging.getLogger("ensemble.py")
print_xenon_path(logger)
env_utils.print(logger)

task_id = env_utils.TASK_ID
trial_ids = env_utils.TRIAL_ID
resource_manager = HttpResourceManager(
    email=env_utils.EMAIL,
    password=env_utils.PASSWORD,
    user_id=env_utils.USER_ID,
    user_token=env_utils.USER_TOKEN
)
# 查询task_id对应的记录
task_records = resource_manager._get_task_records(task_id, resource_manager.user_id)
assert len(task_records) > 0
task_record = task_records[0]
# 机器学习任务类型
ml_task = MLTask(**task_record["ml_task"])
# 实例化一个Xenon对象
kwargs = {
    "resource_manager": resource_manager,
    "log_path": f"{savedpath}/xenon.log",
}
if ml_task.mainTask == "classification":
    xenon = XenonClassifier(**kwargs)
else:
    xenon = XenonRegressor(**kwargs)
# 蹭蹭不进去(is_not_realy_run=True)，只是计算data_manager
xenon.fit(
    X_train=task_record["train_set_id"],
    y_train=task_record["train_label_id"],
    is_not_realy_run=True
)
xenon.ml_task = ml_task
# 开始做集成学习
xenon.estimator = xenon.fit_ensemble(
    task_id=task_id,
    trials_fetcher="GetSpecificTrials",
    trials_fetcher_params={"trial_ids": list(trial_ids)}
)

def can_display_ensemble(xenon):
    if not hasattr(xenon, "ensemble_estimator"):
        return False
    for attr in ["all_score", "confusion_matrix", "weights", "stacked_y_pred"]:
        if not hasattr(xenon.ensemble_estimator, attr):
            return False
    return True

if can_display_ensemble(xenon):
    display(resource_manager, task_id, 100, savedpath, trial_ids=trial_ids,
            ensemble_estimator=xenon.ensemble_estimator, file_name="ensemble_records", output_csv=False,
            xenon=xenon)
# 保存各种ID
save_info_json(
    xenon.experiment_id,
    xenon.task_id,
    getattr(xenon, "hdl_id", None),
    savedpath
)
# 打印id，方便用户在 predict 的时候指定 experiment_id
experiment_id = xenon.experiment_id
logger.info(f"task_id:\t{task_id}")
logger.info(f"experiment_id:\t{experiment_id}")
save_current_expriment_model(savedpath, experiment_id, logger, xenon)
