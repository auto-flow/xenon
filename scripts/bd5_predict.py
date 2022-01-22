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

import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import load, delayed
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline

from scripts.utils import EnvUtils, process_previous_result_dataset, save_info_json, \
    print_xenon_path, load_data_from_datapath
from xenon.metrics import calculate_score
from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.logging_ import setup_logger
from xenon_ext.stack.base import StackEstimator

savedpath = os.getenv("SAVEDPATH", ".")
setup_logger(
    f"{savedpath}/xenon.log"
)
process_previous_result_dataset()
env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/bd5_predict.json")
env_utils.update()
logger = logging.getLogger("bd5_predict.py")
print_xenon_path(logger)
# DATAPATH 有两种形式，
# 1. data/ 文件夹 (传统QSAR模式)
# 2. feature.csv  但此时需要搭配对于列的描述 (用户自定义特征模式)
datapath = os.getenv("DATAPATH")
assert datapath is not None
env_utils.print(logger)

# 查询一条experiment记录，取出task_id
# 下载experiment关联的模型
# 加载数据
# 调用predict方法对数据进行预测
# 保存结果到savedpath
experiment_id = env_utils.EXPERIMENT_ID
######################################
# 实例化resource_manager（资源管理器） #
######################################
resource_manager = HttpResourceManager(
    email=env_utils.EMAIL,
    password=env_utils.PASSWORD,
    user_id=env_utils.USER_ID,
    user_token=env_utils.USER_TOKEN
)
#########################
# 查询一条experiment记录 #
#########################
experiment_records = resource_manager._get_experiment_record(experiment_id)
assert len(experiment_records) > 0, ValueError(f"experiment_id {experiment_id} is invalid.")
experiment_record = experiment_records[0]
task_id = experiment_record.get("task_id", experiment_record.get("task"))
final_model_path = experiment_record["final_model_path"]
logger.info(f"task_id:\t{task_id}")
logger.info(f"experiment_id:\t{experiment_id}")
##########################
# 下载experiment关联的模型 #
##########################
local_path = f"{savedpath}/experiment_{experiment_id}_best_model.bz2"
# 判断非空
assert bool(final_model_path), ValueError(f"experiment {experiment_id}  was not completed normally, is invalid.")
resource_manager.file_system.download(final_model_path, local_path)
xenon_model = load(local_path)
# 单个模型（以Pipeline的形式存在）
if isinstance(xenon_model, Pipeline):
    base_class_list = xenon_model._final_estimator.__class__.__bases__
# 集成模型
elif isinstance(xenon_model, StackEstimator):
    base_class_list = xenon_model.__class__.__bases__
# 系统不能识别的类型
else:
    raise NotImplementedError

if ClassifierMixin in base_class_list:
    model_type = "clf"
    mainTask = "classification"
elif RegressorMixin in base_class_list:
    model_type = "reg"
    mainTask = "regression"
else:
    raise ValueError


def single_dir_predict(cur_datapath):
    ###########
    # 加载数据 #
    ###########
    traditional_qsar_mode = os.path.isdir(cur_datapath)
    feature_name_list = env_utils.FEATURE_NAME_LIST
    column_descriptions = env_utils.COLUMN_DESCRIPTIONS
    train_target_column_name = env_utils.TRAIN_TARGET_COLUMN_NAME
    id_column_name = env_utils.ID_COLUMN_NAME
    data, column_descriptions, _, SMILES = load_data_from_datapath(
        cur_datapath,
        train_target_column_name,
        id_column_name,
        logger,
        traditional_qsar_mode,
        model_type,
        feature_name_list,
        column_descriptions,
        train_set=False
    )
    id_col = None
    if 'id' in column_descriptions:
        id_column_name = column_descriptions['id']
        id_col = data.pop(id_column_name).values
    label = None
    if train_target_column_name and train_target_column_name in data:
        label = data.pop(train_target_column_name).values
    # fixme:写死了xenon_model只能输入DataFrame
    if model_type == "clf":
        pred = xenon_model.predict_proba(data)
    else:
        pred = xenon_model.predict(data)[:, None]

    return id_col, SMILES, label, pred


n_workers = env_utils.N_WORKERS
predict_dir = datapath + "/predict"
if os.path.isdir(predict_dir):
    logger.info(f"开启进程池进行运算， 进程池大小={n_workers}")
    result_pairs = Parallel(backend="loky")(
        delayed(single_dir_predict)(f"{predict_dir}/{sub_dir}")
        for sub_dir in os.listdir(predict_dir)
        if os.path.isdir(f"{predict_dir}/{sub_dir}")
    )
    # for sub_dir in os.listdir(predict_dir): # 用于调试的代码
    #     if os.path.isdir(f"{predict_dir}/{sub_dir}"):
    #         single_dir_predict(f"{predict_dir}/{sub_dir}")
else:
    result_pairs = [single_dir_predict(datapath)]

id_cols, SMILES_list, labels, preds = [], [], [], []
for id_col, SMILES, label, pred in result_pairs:
    if id_col is not None:
        id_cols.append(id_col)
    if label is not None:
        labels.append(label)
    if SMILES is not None:
        SMILES_list.append(SMILES)
    # pred.ndim = 2
    preds.append(pred)

columns = []
values = []
y_true = None
if id_cols:
    columns.append(env_utils.ID_COLUMN_NAME)
    values.append(np.hstack(id_cols)[:, None])
if SMILES_list:
    columns.append("SMILES")
    values.append(np.hstack(SMILES_list)[:, None])
if labels:
    columns.append(env_utils.TRAIN_TARGET_COLUMN_NAME)
    y_true = np.hstack(labels)[:, None]
    values.append(y_true)

concat_pred = np.vstack(preds)
if model_type == "clf":
    values.append(np.argmax(concat_pred, axis=1)[:, None])
    values.append(concat_pred)
    y_pred = concat_pred
    columns.append("RESULT")
    for k in range(concat_pred.shape[1]):
        columns.append(f"PROBA_{k}")
else:
    y_pred = concat_pred.flatten()
    values.append(concat_pred)
    columns.append("RESULT")

df = pd.DataFrame(np.concatenate(values, axis=1), columns=columns)
logger.info(f'完成对predict结果的计算， DataFrame shape = {df.shape}, mem = {df.memory_usage().sum() / 1e6:.2f}M')
df.to_csv(savedpath + "/prediction.csv", index=False)
logger.info(f'predict结果写入到本地')

# 计算在测试集的评价指标
if y_true is not None:
    y_true = y_true.flatten()
    logger.info(f'LABEL存在，开始计算测试集评价指标')
    _, all_scores = calculate_score(
        y_true, y_pred, mainTask,
        metric=None, should_calc_all_metric=True, verbose=True
    )
    columns = ['metric_name', 'metric']
    data = []
    for k, v in all_scores.items():
        data.append([k, v])
    metric_df = pd.DataFrame(data, columns=columns)
    metric_df.to_csv(savedpath + "/test_set_scores.csv", index=False)
    logger.info(f'完成测试集评价指标计算')

save_info_json(
    os.getenv("EXPERIMENT_ID"),
    os.getenv("TASK_ID"),
    os.getenv("HDL_ID"),
    savedpath
)
logger.info('done.')
