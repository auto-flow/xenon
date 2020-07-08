#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import logging

import click

from xenon import XenonClassifier, XenonRegressor
from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.ml_task import MLTask
from scripts.utils import EnvUtils

env_utils = EnvUtils()
env_utils.from_json("env_configs/ensemble.json")
env_utils.update()
env_utils.print()
logger = logging.getLogger("ensemble.py")


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
    ml_task = MLTask(**task_record["ml_task"])
    kwargs = {
        "resource_manager": resource_manager
    }
    if ml_task.mainTask == "classification":
        xenon = XenonClassifier(**kwargs)
    else:
        xenon = XenonRegressor(**kwargs)
    xenon.fit(
        X_train=task_record["train_set_id"],
        y_train=task_record["train_label_id"],
        is_not_realy_run=True
    )
    xenon.estimator = xenon.fit_ensemble(
        task_id=task_id,
        trials_fetcher="GetSpecificTrials",
        trials_fetcher_params={"trial_ids": list(trial_ids)}
    )
    # 获取与task_id关联的所有experiment记录，找到experiment_type!=ensemble的记录，下载data_manager


if __name__ == '__main__':
    main()
