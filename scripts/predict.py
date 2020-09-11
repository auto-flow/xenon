#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
###################################
# Insert current path to sys path #
###################################
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
###################################
import logging
import os

import pandas as pd
from joblib import load

from xenon.resource_manager.http import HttpResourceManager
from xenon.utils.logging_ import setup_logger
from scripts.utils import EnvUtils, load_data_from_datapath, process_previous_result_dataset, save_info_json, \
    print_xenon_path

savedpath = os.getenv("SAVEDPATH", ".")
setup_logger(
    f"{savedpath}/xenon.log"
)
process_previous_result_dataset()
env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/predict.json")
env_utils.update()
logger = logging.getLogger("predict.py")
print_xenon_path(logger)
# DATAPATH 有两种形式，
# 1. data/ 文件夹 (传统QSAR模式)
# 2. feature.csv  但此时需要搭配对于列的描述 (用户自定义特征模式)
datapath = os.getenv("DATAPATH")
assert datapath is not None

logger.info(f"DATAPATH: {datapath}")
traditional_qsar_mode = True
if os.path.isdir(datapath):
    print("DATAPATH 为传统QSAR模式，传入的是分子指纹矢量化后的结果。")
else:
    traditional_qsar_mode = False
    print("DATAPATH 为用户自定义模式，传入的是用户自定义的特征文件。")
#     print("""需要注意的是，该模式下用户需要指定'COLUMN_DESCRIPTIONS'环境变量
# 如：
# COLUMN_DESCRIPTIONS = {'id' : "NAME" ,'target' : 'pIC50','ignore' : ['SMILES']}
# """)
logger.info(f"traditional_qsar_mode = {traditional_qsar_mode}")
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
xenon = load(local_path)
###########
# 加载数据 #
###########
if os.path.exists(f"{datapath}/predict"):
    Path(f"{savedpath}/prediction").mkdir()
    for sub_datapath in Path(f"{datapath}/predict").iterdir():
        feature_name_list = env_utils.FEATURE_NAME_LIST
        column_descriptions = {}
        train_target_column_name = None
        id_column_name = None
        model_type = None
        data, _ = load_data_from_datapath(
            sub_datapath,
            train_target_column_name,
            id_column_name,
            logger,
            traditional_qsar_mode,
            model_type,
            feature_name_list,
            column_descriptions
        )
        ###############################
        # 调用predict方法对数据进行预测 #
        ###############################
        if xenon.estimator is None:
            xenon.estimator = xenon.ensemble_estimator
        result = xenon.predict(data)
        # 把ID与result拼在一起
        test_id_seq = getattr(xenon.data_manager, "test_id_seq", None)
        df = {
            "result": result
        }
        if test_id_seq is not None:
            df.update({
                "ID": test_id_seq,
            })
        df = pd.DataFrame(df)
        ######################
        # 保存结果到savedpath #
        ######################
        predict_path = f"{savedpath}/prediction/{sub_datapath.name}.csv"
        df.to_csv(predict_path, index=False)
    first = True
    prediction_file = None
    for sub_result in Path(f"{savedpath}/prediction").iterdir():
        if first:
            prediction_file = pd.read_csv(sub_result, header=0)
            first = False
        else:
            tmp_file = pd.read_csv(sub_result, header=0)
            prediction_file = prediction_file.append(tmp_file)
    if not first:
        prediction_file.to_csv(f"{savedpath}/prediction.csv", index=False)
else:
    feature_name_list = env_utils.FEATURE_NAME_LIST
    column_descriptions = {}
    train_target_column_name = None
    id_column_name = None
    model_type = None
    data, _ = load_data_from_datapath(
        datapath,
        train_target_column_name,
        id_column_name,
        logger,
        traditional_qsar_mode,
        model_type,
        feature_name_list,
        column_descriptions
    )
    ###############################
    # 调用predict方法对数据进行预测 #
    ###############################
    if xenon.estimator is None:
        xenon.estimator = xenon.ensemble_estimator
    result = xenon.predict(data)
    # 把ID与result拼在一起
    test_id_seq = getattr(xenon.data_manager, "test_id_seq", None)
    df = {
        "result": result
    }
    if test_id_seq is not None:
        df.update({
            "ID": test_id_seq,
        })
    df = pd.DataFrame(df)
    ######################
    # 保存结果到savedpath #
    ######################
    predict_path = f"{savedpath}/prediction.csv"
    df.to_csv(predict_path, index=False)
save_info_json(
    os.getenv("EXPERIMENT_ID"),
    os.getenv("TASK_ID"),
    os.getenv("HDL_ID"),
    savedpath
)