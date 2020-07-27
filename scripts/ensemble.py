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

import click

from xenon.utils.logging_ import setup_logger
from xenon import XenonClassifier, XenonRegressor
from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.ml_task import MLTask
from scripts.utils import EnvUtils, save_current_expriment_model, save_info_json

env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/ensemble.json")
env_utils.update()
logger = logging.getLogger("ensemble.py")
savedpath = os.getenv("SAVEDPATH", ".")
setup_logger(
    f"{savedpath}/xenon.log"
)
env_utils.print(logger)


@click.command()
@click.argument("trial_ids", nargs=-1)
@click.option("--task_id")
def main(task_id, trial_ids):
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
    # 查询输入的trials记录，并打印模型表现
    logger.info("Selected trials' performance:")
    default_metric = "accuracy" if ml_task.mainTask == "classification" else "r2"
    for trial_id in trial_ids:
        trial_record = resource_manager._get_trial_records_by_id(trial_id, task_id, 0)
        performance = trial_record["all_score"][default_metric]
        logger.info(f"trial_id = {trial_id}\t{default_metric} = {performance}")
    # 蹭蹭不进去(is_not_realy_run=True)，只是计算data_manager
    xenon.fit(
        X_train=task_record["train_set_id"],
        y_train=task_record["train_label_id"],
        is_not_realy_run=True
    )
    # 开始做集成学习
    xenon.estimator = xenon.fit_ensemble(
        task_id=task_id,
        trials_fetcher="GetSpecificTrials",
        trials_fetcher_params={"trial_ids": list(trial_ids)}
    )
    # 保存各种ID
    save_info_json(
        xenon.experiment_id,
        xenon.task_id,
        getattr(xenon,"hdl_id"),
        savedpath
    )
    # 打印id，方便用户在 predict 的时候指定 experiment_id
    experiment_id = xenon.experiment_id
    logger.info(f"task_id:\t{task_id}")
    logger.info(f"experiment_id:\t{experiment_id}")
    save_current_expriment_model(savedpath, experiment_id, logger, xenon)


if __name__ == '__main__':
    main()
