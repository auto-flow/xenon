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

from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.logging_ import setup_logger
from scripts.utils import EnvUtils, display, process_previous_result_dataset, save_info_json, print_xenon_path

savedpath = os.getenv("SAVEDPATH", ".")
setup_logger(
    f"{savedpath}/xenon.log"
)
process_previous_result_dataset()
env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/display.json")
env_utils.update()
logger = logging.getLogger("display.py")
print_xenon_path(logger)
env_utils.print(logger)

task_id = env_utils.TASK_ID
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
display_size = env_utils.DISPLAY_SIZE
display(resource_manager, task_id, display_size, savedpath)
save_info_json(
    os.getenv("EXPERIMENT_ID"),
    os.getenv("TASK_ID"),
    os.getenv("HDL_ID"),
    savedpath
)
