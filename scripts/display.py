#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click

from xenon.resource_manager.http import HttpResourceManager
from scripts.utils import EnvUtils

env_utils = EnvUtils()
env_utils.from_json("env_configs/display.json")
env_utils.update()
env_utils.print()
logger = logging.getLogger("display.py")


@click.command()
@click.option("--task_id")
def main(task_id):
    assert task_id is not None, ValueError("task_id is None")
    ######################################
    # 实例化resource_manager（资源管理器） #
    ######################################
    resource_manager = HttpResourceManager(
        email=env_utils.EMAIL,
        password=env_utils.PASSWORD,
        user_id=env_utils.USER_ID,
        user_token=env_utils.USER_TOKEN
    )
    user_id = resource_manager.user_id
    records = resource_manager._get_sorted_trial_records(task_id, user_id, env_utils.DISPLAY_SIZE)
    ml_task, y_train = resource_manager.get_ensemble_needed_info(task_id)
    y_train = y_train.data
    # 处理records, 加载y_info_path
    processed_records = []
    for record in records:
        y_info_path = record["y_info_path"]
        # keys: ['y_true_indexes', 'y_preds', 'y_test_pred']
        y_info = resource_manager.file_system.load_pickle(y_info_path)
        record["y_info"] = y_info
        processed_records.append(record)
    # todo: 对接 mock_display.py
    import pickle
    mock_data = {
        "mainTask": ml_task.mainTask,
        "records": processed_records,
        "y_train": y_train
    }
    Path("mock_data.pkl").write_bytes(pickle.dumps(mock_data))


if __name__ == '__main__':
    # 接受参数：task_id
    main()
