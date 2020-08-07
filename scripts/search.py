#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
###################################
# Insert current path to sys path #
###################################
import os
import sys
from typing import Union, Optional

import psutil

sys.path.insert(0, os.getcwd())
###################################
import logging
import multiprocessing as mp
import numpy as np
from xenon.utils.logging_ import setup_logger
from xenon import XenonClassifier, XenonRegressor
from xenon.hdl.hdl_constructor import HDL_Constructor
from xenon.resource_manager.http import HttpResourceManager
from xenon.tuner import Tuner
from scripts.utils import EnvUtils, save_current_expriment_model, load_data_from_datapath, display, save_info_json, \
    print_xenon_path


def search(datapath: Optional[str] = None, save_in_savedpath=True) -> Union[XenonClassifier, XenonRegressor]:
    '''
    Search Stage , Input TrainSet DataFrame path , Output trained model

    Parameters
    ----------
    datapath: str or None
    TrainSet DataFrame path. if is None, load  DATAPATH  env

    Returns
    -------
    traned model: XenonClassifier or XenonRegressor

    '''
    env_utils = EnvUtils()
    env_utils.from_json("env_configs/common.json")
    env_utils.from_json("env_configs/search.json")
    # env_utils.from_json("env_configs/display.json")
    env_utils.update()
    logger = logging.getLogger("search.py")
    print_xenon_path(logger)
    # DATAPATH 有两种形式，
    # 1. data/ 文件夹 (传统QSAR模式)
    # 2. feature.csv  但此时需要搭配对于列的描述 (用户自定义特征模式)
    if datapath is None:
        datapath = os.getenv("DATAPATH")
    savedpath = os.getenv("SAVEDPATH", ".")
    assert bool(datapath), ValueError(f"Search Stage must has a dataset!")
    setup_logger(
        f"{savedpath}/xenon.log"
    )
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

    ######################################
    # 实例化resource_manager（资源管理器） #
    ######################################
    resource_manager = HttpResourceManager(
        email=env_utils.EMAIL,
        password=env_utils.PASSWORD,
        user_id=env_utils.USER_ID,
        user_token=env_utils.USER_TOKEN
    )
    #######################
    # 实例化tuner（优化器） #
    #######################
    search_thread_num = env_utils.SEARCH_THREAD_NUM
    n_jobs_in_algorithm = env_utils.N_JOBS_IN_ALGORITHM
    logger.info(f"Target Computer's CPU count = {mp.cpu_count()}")
    if n_jobs_in_algorithm is None:
        logger.info(
            "N_JOBS_IN_ALGORITHM is None, will calc n_jobs_in_algorithm by 'mp.cpu_count() // search_thread_num'")
        n_jobs_in_algorithm = mp.cpu_count() // search_thread_num
    logger.info(f"n_jobs_in_algorithm = {n_jobs_in_algorithm}")
    per_run_time_limit = env_utils.PER_RUN_TIME_LIMIT
    logger.info(f"per_run_time_limit = {per_run_time_limit // 60} minutes")
    per_run_memory_limit = env_utils.PER_RUN_MEMORY_LIMIT
    m = psutil.virtual_memory()
    total = m.total / 1024 / 1024
    free = m.free / 1024 / 1024
    used = m.used / 1024 / 1024
    logger.info(f"Target Computer's Memory Info: total = {total:.2f}M, free = {free:.2f}M, used = {used:.2f}M")
    if per_run_memory_limit is None:
        logger.info("PER_RUN_MEMORY_LIMIT is None, will calc per_run_memory_limit by 'total / search_thread_num'")
        per_run_memory_limit = total / search_thread_num
    logger.info(f"per_run_memory_limit = {per_run_memory_limit}M")
    tuner = Tuner(
        initial_runs=env_utils.RANDOM_RUNS,
        run_limit=env_utils.BAYES_RUNS,
        n_jobs=search_thread_num,
        per_run_time_limit=per_run_time_limit,
        per_run_memory_limit=per_run_memory_limit,
        n_jobs_in_algorithm=n_jobs_in_algorithm
    )
    ############################################
    # 实例化hdl_constructor（超参描述语言构造器） #
    ############################################
    model_type = env_utils.MODEL_TYPE
    if model_type == "clf":
        DAG_workflow = env_utils.CLF_WORKFLOW
    else:
        DAG_workflow = env_utils.REG_WORKFLOW
    hdl_constructor = HDL_Constructor(
        DAG_workflow=DAG_workflow
    )
    #####################
    # 实例化Xenon对象 #
    #####################
    random_state = env_utils.RANDOM_STATE
    if random_state is None:
        logger.info("RANDOM_STATE is None, will choose a random_state between 0 and 10000.")
        random_state = np.random.randint(0, 10000)
    logger.info(f"random_state = {random_state}")
    kwargs = {
        "tuner": tuner,
        "hdl_constructor": hdl_constructor,
        "resource_manager": resource_manager,
        "log_path": f"{savedpath}/xenon.log",
        "random_state": random_state
    }
    if model_type == "clf":
        xenon = XenonClassifier(**kwargs)
    else:
        xenon = XenonRegressor(**kwargs)
    ######################
    # 从DATAPATH中加载数据 #
    ######################
    feature_name_list = env_utils.FEATURE_NAME_LIST
    column_descriptions = env_utils.COLUMN_DESCRIPTIONS
    train_target_column_name = env_utils.TRAIN_TARGET_COLUMN_NAME
    id_column_name = env_utils.ID_COLUMN_NAME
    # 公用的数据加载部分
    data, column_descriptions = load_data_from_datapath(
        datapath,
        train_target_column_name,
        id_column_name,
        logger,
        traditional_qsar_mode,
        model_type,
        feature_name_list,
        column_descriptions
    )
    #######################################
    # 调用Xenon对象的fit函数启动搜索过程  #
    #######################################
    # ------------------------------#
    # specific_task_token 与 元数据  #
    # ------------------------------#
    specific_task_token = env_utils.SPECIFIC_TASK_TOKEN
    if specific_task_token is None:
        specific_task_token = ""
    dataset_metadata = {}
    if env_utils.DATASET_NAME is not None:
        dataset_metadata.update({"dataset_name": env_utils.DATASET_NAME})
    if env_utils.DATASET_DESCRIPTION is not None:
        dataset_metadata.update({"dataset_description": env_utils.DATASET_DESCRIPTION})
    task_metadata = {}
    if env_utils.TASK_NAME is not None:
        task_metadata.update({"task_name": env_utils.TASK_NAME})
    if env_utils.TASK_DESCRIPTION is not None:
        task_metadata.update({"task_description": env_utils.TASK_DESCRIPTION})
    # -----------------#
    # metrics 评价指标  #
    # -----------------#
    metric = env_utils.METRIC
    # clf metrics
    from xenon.metrics import accuracy, mcc, sensitivity, specificity, \
        balanced_accuracy, f1, precision, recall, pac_score
    # reg metrics
    from xenon.metrics import r2, mean_squared_error, median_absolute_error

    metrics = {
        # clf
        "accuracy": accuracy,
        "mcc": mcc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pac_score": pac_score,
        # reg
        "r2": r2,
        "mean_squared_error": mean_squared_error,
        "median_absolute_error": median_absolute_error,
    }
    if metric is not None:
        if metric not in metrics:
            logger.warning(f"metric '{metric}' is invalid. Valid metrics are '{list(metrics.keys())}'")
            metric = None  # use default metric
        metric = metrics[metric]
    # ------------------#
    # splitter 数据切分  #
    # ------------------#
    from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit

    kfold = env_utils.KFOLD
    if kfold > 1:
        kfold = int(kfold)
        kfold_kwargs = dict(n_splits=kfold, shuffle=True, random_state=0)
        if model_type == "clf":
            splitter = StratifiedKFold(**kfold_kwargs)
        else:
            splitter = KFold(**kfold_kwargs)
    elif kfold == 1:
        splitter = LeaveOneOut()
    elif kfold < 1:
        splitter = ShuffleSplit(n_splits=1, test_size=kfold, random_state=0)
    else:
        raise ValueError(f"Invalid KFOLD {kfold}.")
    logger.info(f"KFOLD = {kfold}, data_splitter will be interpret as {splitter}.")

    # ------------------------#
    # stacking 集成学习的参数  #
    # ------------------------#
    ensemble_size = env_utils.ENSEMBLE_SIZE
    if ensemble_size <= 1:
        logger.info("Xenon will not do stacking.")
        fit_ensemble_params = False
    else:
        logger.info(f"Xenon will stack {ensemble_size} models.")
        fit_ensemble_params = {"trials_fetcher_params": {"k": ensemble_size}}
    # --------------#
    # 启动Xenon  #
    # --------------#
    xenon.fit(
        X_train=data,
        column_descriptions=column_descriptions,
        metric=metric,
        splitter=splitter,
        specific_task_token=specific_task_token,
        dataset_metadata=dataset_metadata,
        task_metadata=task_metadata,
        fit_ensemble_params=fit_ensemble_params
    )
    if save_in_savedpath:
        # 保存各种ID
        save_info_json(
            xenon.experiment_id,
            xenon.task_id,
            xenon.hdl_id,
            savedpath
        )
    ######################################
    # 实验完成，保存最好的模型到SAVEDPATH  #
    ######################################
    experiment_id = xenon.experiment_id
    if save_in_savedpath:
        save_current_expriment_model(savedpath, experiment_id, logger, xenon)
    ###########################
    # 调用display.py进行可视化 #
    ###########################
    if save_in_savedpath:
        display(resource_manager, xenon.task_id,
                env_utils.DISPLAY_SIZE, savedpath)
    return xenon


if __name__ == '__main__':
    search(datapath=None, save_in_savedpath=True)
