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
os.system("mkdir -p $SAVEDPATH")
# fixme: 用于解决xlearn在xenon镜像中的问题
# https://github.com/aksnzhy/xlearn/issues/215
os.environ["USER"] = "test"
###################################
import logging
import os

from scripts.utils import EnvUtils, save_info_json, process_previous_result_dataset, \
    print_xenon_path, save_current_expriment_model_xenon4bigdata
from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.logging_ import setup_logger
from xenon4bigdata.ensemble import get_ensemble_model_by_trial_ids
from xenon4bigdata.external_delivery._external_delivery_process import external_delivery

savedpath = os.getenv("SAVEDPATH", ".")
setup_logger(
    f"{savedpath}/xenon.log"
)
process_previous_result_dataset()
env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/bd4_ensemble.json")
env_utils.update()
logger = logging.getLogger("bd4_ensemble.py")
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
# 创建experiment记录
experiment_id = resource_manager._insert_experiment_record(
    user_id=0, hdl_id="0", task_id=task_id, experiment_type="ensemble_modeling", experiment_config={},
    additional_info={}
)
# 打印id，方便用户在 predict 的时候指定 experiment_id
logger.info(f"task_id:\t{task_id}")
logger.info(f"experiment_id:\t{experiment_id}")
# 查询task_id对应的记录
task_records = resource_manager._get_task_records(task_id, resource_manager.user_id)
assert len(task_records) > 0
task_record = task_records[0]
# 调用和search一致的ensemble构建函数（提炼出了一个可复用的函数）
stacker = get_ensemble_model_by_trial_ids(resource_manager, task_id, trial_ids, None)
logger.info(str(stacker))
# 保存各种ID
save_info_json(
    experiment_id,
    task_id,
    "0",
    savedpath
)
# fixme: 暂时还没有可视化，报表
# 保存实验（ensemble）产出的模型，并上传数据库
save_current_expriment_model_xenon4bigdata(savedpath, experiment_id, logger, stacker, resource_manager)
# 对外交付
external_delivery(models=stacker.estimators_list)
logger.info('done.')
