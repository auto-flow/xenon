#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
###################################
# Insert current path to sys path #
###################################
import os
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import wait
from pathlib import Path

from xenon.ensemble.mean.regressor import MeanRegressor
from xenon.ensemble.stack.base import StackEstimator
from xenon.ensemble.vote.classifier import VoteClassifier

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
is_classifier = ("Classifier" in xenon.__class__.__name__)


def single_dir_predict(data_path, saved_dir, saved_in_dir=False):
    ###########
    # 加载数据 #
    ###########
    feature_name_list = env_utils.FEATURE_NAME_LIST
    column_descriptions = {}
    train_target_column_name = None
    id_column_name = None
    model_type = None
    data, _ = load_data_from_datapath(
        data_path,
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

    # baseline
    if saved_in_dir:
        complex_predict(data, f"{saved_dir}/prediction", saved_filename=sub_datapath.name)
    else:
        complex_predict(data, saved_dir)
    # 花架子
    if isinstance(xenon.estimator, StackEstimator) and getattr(xenon, "ensemble_info", None) is not None:
        logger.info("current model is a stacking model, will use every base-model to do prediction.")
        xenon.output_ensemble_info()
        estimators_list = xenon.estimator.estimators_list
        pre_estimator = xenon.estimator
        for i, trial_id in enumerate(xenon.trial_ids):
            estimators = estimators_list[i]
            if is_classifier:
                new_ensemble = VoteClassifier(estimators)
            else:
                new_ensemble = MeanRegressor(estimators)
            xenon.estimator = new_ensemble
            if saved_in_dir:
                complex_predict(data, f"{saved_dir}/prediction_{trial_id}", saved_filename=sub_datapath.name)
            else:
                complex_predict(data, saved_dir, suffix=f"_{trial_id}")
        xenon.estimator = pre_estimator


def complex_predict(data, saved_dir, saved_filename="prediction", suffix=""):
    result = xenon.predict(data)
    # 把ID与result拼在一起
    test_id_seq = getattr(xenon.data_manager, "test_id_seq", None)
    df = {}
    if test_id_seq is not None:
        df.update({
            "ID": test_id_seq,
        })
    df["RESULT"] = result
    df = pd.DataFrame(df)
    if is_classifier:
        proba_ = xenon.predict_proba(data)
        proba = pd.DataFrame(proba_, columns=[f"PROBA_{i}" for i in range(proba_.shape[1])])
    else:
        proba = pd.DataFrame()
    df = pd.concat([df, proba], axis=1)
    predict_path = f"{saved_dir}/{saved_filename}{suffix}.csv"
    df.to_csv(predict_path, index=False)


def combine_csv_in_dir(dir):
    first = True
    prediction_file = None
    for sub_result in Path(dir).iterdir():
        if first:
            prediction_file = pd.read_csv(sub_result, header=0)
            first = False
        else:
            tmp_file = pd.read_csv(sub_result, header=0)
            prediction_file = prediction_file.append(tmp_file)
    if not first:
        prediction_file.to_csv(f"{dir}.csv", index=False)


if os.path.exists(f"{datapath}/predict"):
    # 切分数据批量预测
    Path(f"{savedpath}/prediction").mkdir(exist_ok=True)
    if isinstance(xenon.estimator, StackEstimator) and getattr(xenon, "ensemble_info", None) is not None:
        for trial_id in xenon.trial_ids:
            Path(f"{savedpath}/{trial_id}").mkdir(exist_ok=True)

    thread_num = os.getenv("PREDICT_THREAD_NUM")
    if thread_num is not None and int(thread_num) > 1:
        thread_num = int(thread_num)
        with ThreadPoolExecutor(max_workers=thread_num) as t:
            task_list = []
            for sub_datapath in Path(f"{datapath}/predict").iterdir():
                task = t.submit(single_dir_predict, sub_datapath, savedpath, True)
                task_list.append(task)
                wait(task_list)
    else:
        for sub_datapath in Path(f"{datapath}/predict").iterdir():
            single_dir_predict(sub_datapath, savedpath, saved_in_dir=True)

    combine_csv_in_dir(f"{savedpath}/prediction")
    if isinstance(xenon.estimator, StackEstimator) and getattr(xenon, "ensemble_info", None) is not None:
        for trial_id in xenon.trial_ids:
            combine_csv_in_dir(f"{savedpath}/prediction_{trial_id}")
else:
    single_dir_predict(datapath, savedpath)

save_info_json(
    os.getenv("EXPERIMENT_ID"),
    os.getenv("TASK_ID"),
    os.getenv("HDL_ID"),
    savedpath
)
