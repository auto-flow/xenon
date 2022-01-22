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
from pathlib import Path
from time import time

import json5
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.utils.multiclass import type_of_target
from uuid import uuid4

import xenon4bigdata
from neutron import fmin
from neutron.hdl import hdl2cs
from neutron.optimizer import SMACOptimizer
from scripts.utils import EnvUtils, display, save_info_json, \
    print_xenon_path, save_current_expriment_model_xenon4bigdata, parse_env_estimator_params
from xenon.constants import binary_classification_task, multiclass_classification_task, regression_task
from xenon.data_container import NdArrayContainer
from xenon.metrics.get_metric_dict import *
from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.dict_ import object_kwargs2dict
from xenon.utils.hash import get_file_md5, get_hash_of_str
from xenon.utils.logging_ import setup_logger
from xenon4bigdata.evaluator import Evaluator
from xenon4bigdata.ensemble import get_ensemble_model_by_trial_ids
from xenon4bigdata.external_delivery._external_delivery_process import external_delivery

env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/bd2_search.json")
# env_utils.from_json("env_configs/display.json")
env_utils.update()
logger = logging.getLogger("bd2_search.py")
print_xenon_path(logger)
# DATAPATH 有两种形式，
# 1. data/ 文件夹 (传统QSAR模式)
# 2. feature.csv  但此时需要搭配对于列的描述 (用户自定义特征模式)
datapath = os.getenv("DATAPATH")
savedpath = os.getenv("SAVEDPATH", ".")
Path(savedpath).mkdir(parents=True, exist_ok=True)
assert bool(datapath), ValueError(f"Search Stage must has a dataset!")
setup_logger(
    f"{savedpath}/xenon.log"
)
logger.info(f"DATAPATH: {datapath}")
env_utils.print(logger)

############################
# 为了解决DATAPATH只读的问题 #
############################
# xlearn 遇到这样的问题
# 所以如果候选集里面没有xlearn的话，不用做
has_xlearn = False
for candidate in env_utils.ESTIMATOR_CHOICES:
    if "xl" in candidate:
        has_xlearn = True
        logger.info(f"has_xlearn = True,  解决DATAPATH只读的问题")
        break
if has_xlearn:
    start_time = time()
    tmp_datapath = '/tmp/datapath'
    os.system(f'rm -rf {tmp_datapath}')
    os.system(f'mkdir -p {tmp_datapath}')
    os.system(f'cp $DATAPATH/* {tmp_datapath}/')
    os.environ['DATAPATH'] = tmp_datapath
    cost_time = time() - start_time
    logger.info(f"copy datapath cost {cost_time:.2f}s")

######################################
# 实例化resource_manager（资源管理器） #
######################################
resource_manager = HttpResourceManager(
    email=env_utils.EMAIL,
    password=env_utils.PASSWORD,
    user_id=env_utils.USER_ID,
    user_token=env_utils.USER_TOKEN
)
# ------------------------------#
# 根据model_type和label判断task  #
# ------------------------------#
model_type = env_utils.MODEL_TYPE
y = pd.read_csv(datapath + "/LABEL.csv")["LABEL"].values
if model_type == "reg":
    ml_task = regression_task
else:
    target_type = type_of_target(y)
    if target_type == "binary":
        ml_task = binary_classification_task
    elif target_type == "multiclass":
        ml_task = multiclass_classification_task
    else:
        raise ValueError(f"Invalid target_type {target_type}!")
# -----------------#
# metrics 评价指标  #
# -----------------#
metric = env_utils.METRIC

# 默认为None，根据任务类型，分类mcc，回归r2
if metric is None:
    if ml_task.mainTask == "classification":
        metric: Scorer = mcc
    else:
        metric: Scorer = r2
# 如果用户自定义，就从字典里面根据字符串取
else:
    if metric not in metric_dict:
        raise ValueError(f"metric '{metric}' is invalid. Valid metrics are '{list(metric_dict.keys())}'")
    metric: Scorer = metric_dict[metric]

#############################################
# 调用 resource_manager 进行本次实验的数据记录 #
#############################################
# ---------------------------------------------
# --------------#
# task id 记录  #
# --------------#
# 读一个小文件的md5来感知任务id
# 目的：用户跑【同一个数据】的时候可以热启动，可以在上一次的checkpoint上继续跑

file_md5 = get_file_md5(datapath + "/valid_0.txt")
specific_task_token = str(env_utils.SPECIFIC_TASK_TOKEN)
y_container = NdArrayContainer("TrainLabel", dataset_instance=y, resource_manager=resource_manager)
train_label_id = y_container.get_hash()
y_container.upload()
# 修改specific_task_token等于修改task_id
# 目的：修改SPECIFIC_TASK_TOKEN破坏热启动，建立一个新的checkpoint
columns_txt = Path(os.environ['DATAPATH'] + "/columns.txt").read_text()
task_id = get_hash_of_str(
    file_md5 + str(env_utils.ESTIMATOR_CHOICES)+ specific_task_token)
logger.info(f"specific_task_token = {specific_task_token}")
resource_manager._insert_task_record(
    task_id=task_id, user_id=0, metric=metric.name, splitter={},
    ml_task=object_kwargs2dict(ml_task, func="__new__", keys=ml_task._fields),
    train_set_id="", test_set_id="", train_label_id=train_label_id, test_label_id="",
    specific_task_token=specific_task_token,
    task_metadata={}, sub_sample_indexes=[],
    sub_feature_indexes=[]
)
# -------------#
# hdl id 记录  #
# -------------#
# fixme: hdl这个表废弃，hdl_id默认为0
hdl_id = "0"
resource_manager._insert_hdl_record(
    task_id="0", hdl_id=hdl_id, user_id=0, hdl={}, hdl_metadata={}
)
# --------------------#
# experiment id 记录  #
# --------------------#
# search的上游数据是preprocess，读取上游构造和存储的transformers，上传到
transformers_local_path = f"{datapath}/transformers.pkl"
nitrogen_dataset_name = f"transformers/{uuid4().hex}.pkl"  # fixme： 直接搞成uuid，相当于每次启动search都会传一遍transformers
transformers = load(transformers_local_path)
transformers_dataset_id = resource_manager.file_system.upload(
    nitrogen_dataset_name,
    transformers_local_path)
experiment_id = resource_manager._insert_experiment_record(
    user_id=0, hdl_id=hdl_id, task_id=task_id, experiment_type="auto_modeling", experiment_config={},
    additional_info={
        "transformers_dataset_id": transformers_dataset_id
    }
)
# ----------------#
# 打印保存 各种ID  #
# ----------------#
# 保存各种ID
save_info_json(
    experiment_id,
    task_id,
    hdl_id,
    savedpath
)
logger.info(f"task_id\t=\t{task_id}")
logger.info(f"hdl_id\t=\t{hdl_id}")
logger.info(f"experiment_id\t=\t{experiment_id}")
resource_manager.task_id = task_id
resource_manager.hdl_id = hdl_id
resource_manager.experiment_id = experiment_id
################################################
# 调用优化框架neutron 对评价函数evaluator 进行优化 #
################################################
# bd1_search.py 的目的是启动一个BO(贝叶斯优化)循环，然后产出一个建立在xenon_ext基础上的模型
# 这个模型甚至可以直接对外交付
# -------------------------------------------------
# 热启动（load之前搜出来的checkpoint，用来启动BO）
if env_utils.WARM_START:
    # 取top20个就好了。todo： 将硬编码的top20提炼为环境变量
    records = resource_manager._get_sorted_trial_records(task_id, 0, limit=20)
    initial_points = []
    for record in records:
        initial_points.append(record['dict_hyper_param'])
else:
    initial_points = None
# 首先定义评价函数
# 固定的 学习器参数： estimator_params
estimator_params = parse_env_estimator_params(env_utils.ESTIMATOR_PARAMS)
logger.info(f"estimator_params = {estimator_params}")
evaluator = Evaluator(ml_task, metric, resource_manager, estimator_params)
# 然后定义超参空间 todo: 用户自定义hdl.json
hdl_json_path = Path(xenon4bigdata.__file__).parent.as_posix() + "/hdl.json"
with open(hdl_json_path) as f:
    hdl = json5.load(f)
estimator_choices = env_utils.ESTIMATOR_CHOICES
hdl_estimator: dict = hdl['estimator(choice)']
current_estimators = list(hdl_estimator.keys())
invalid_estimators = set(current_estimators) - set(estimator_choices)
for e in invalid_estimators:
    hdl_estimator.pop(e)
cs = hdl2cs(hdl)
optimizer = SMACOptimizer(min_points_in_model=env_utils.N_RANDOM_STARTS)
n_workers = env_utils.N_WORKERS
random_state = np.random.randint(1, 100000)
logger.info(f"random_state = {random_state}")
result = fmin(
    eval_func=evaluator, config_space=cs, optimizer="SMAC",
    n_iterations=env_utils.N_ITERATIONS, n_jobs=n_workers, random_state=random_state,
    initial_points=initial_points)
dump(result, savedpath + "/opt_result.pkl")
#############################################################
# 读数据库，加载一个最好的模型，然后组装出本次实验产生的final模型 #
#############################################################
'''
目前有两种做集成学习（stacking）的方法：
1. 根据当前task_id，选取loss最小的k个trial_id  
2. 用户指定trial_ids，并且这些trial_ids可以task_id不同（但是训练数据需要是一致的）
search阶段模型产生时采用方案1，fetch当前task_id下最好的trial_id对应的模型
'''
trial_ids = resource_manager._get_best_k_trial_ids(
    task_id, 0, k=1
)
final_model = get_ensemble_model_by_trial_ids(resource_manager, task_id, trial_ids, None)
######################################
# 实验完成，保存最好的模型到SAVEDPATH   #
######################################
display(resource_manager, task_id,
        env_utils.DISPLAY_SIZE, savedpath)
save_current_expriment_model_xenon4bigdata(savedpath, experiment_id, logger, final_model, resource_manager)
# 对外交付
external_delivery([final_model])
logger.info("done.")
