#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
###################################
# Insert current path to sys path #
###################################
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
###################################
from typing import Union, Optional

import psutil
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit
import logging
import multiprocessing as mp
import numpy as np
from xenon.ensemble.stack.base import StackEstimator
from xenon.utils.logging_ import setup_logger
from xenon import XenonClassifier, XenonRegressor
from xenon.hdl.hdl_constructor import HDL_Constructor
from xenon.resource_manager.http import HttpResourceManager
from xenon.tuner import Tuner
from scripts.utils import EnvUtils, save_current_expriment_model, load_data_from_datapath, display, save_info_json, \
    print_xenon_path
from mlxtend.evaluate import PredefinedHoldoutSplit


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
    Path(savedpath).mkdir(parents=True, exist_ok=True)
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
    # 基于方差的特征选择
    use_var_feature_selector = env_utils.USE_VAR_FEATURE_SELECTOR
    # 特征变换器
    scaler = env_utils.SCALER
    # 基于模型的特征选择
    feature_selector = env_utils.FEATURE_SELECTOR
    # 分解器（PCA降维或者KernelPCA增维）
    decomposer = env_utils.DECOMPOSER
    # 分箱器
    discretizer = env_utils.DISCRETIZER
    # 分类器
    classifier = env_utils.CLASSIFIER
    # 回归器
    regressor = env_utils.REGRESSOR
    # start parse
    pre_feat_group = "num"
    table = [
        ["var_selected", use_var_feature_selector, "select.variance"],
        ["scaled", scaler, "scale.flexible"],
        ["model_selected", feature_selector, "select.flexible"],
        ["decomposed", decomposer, "decompose.flexible"],
        ["discretized", discretizer, "discretize.flexible"],
    ]

    DAG_workflow = collections.OrderedDict()
    for feat_group, candidates, module in table:
        assert isinstance(candidates, (bool, list, tuple)), ValueError
        if candidates == False:
            continue
        if isinstance(candidates, (list, tuple)) and \
                (len(candidates) == 0 or (len(candidates) == 1 and candidates[0] == "none")):
            continue
        key = f"{pre_feat_group}->{feat_group}"
        if candidates == True:
            DAG_workflow[key] = module
        else:
            DAG_workflow[key] = {
                "_name": module,
                "strategy": {"_type": "choice", "_value": candidates}
            }
        pre_feat_group = feat_group
    DAG_workflow[f"{pre_feat_group}->target"] = classifier if model_type == "clf" else regressor
    DAG_workflow = dict(DAG_workflow)
    # end parse
    hdl_constructor = HDL_Constructor(
        DAG_workflow=DAG_workflow
    )
    ######################
    # 从DATAPATH中加载数据 #
    ######################
    feature_name_list = env_utils.FEATURE_NAME_LIST
    # column_descriptions = env_utils.COLUMN_DESCRIPTIONS
    ignore_columns = env_utils.IGNORE_COLUMNS
    train_target_column_name = env_utils.TRAIN_TARGET_COLUMN_NAME
    id_column_name = env_utils.ID_COLUMN_NAME
    # 公用的数据加载部分（SPLIT表示自定义切分）
    data, column_descriptions, SPLIT, _ = load_data_from_datapath(
        datapath,
        train_target_column_name,
        id_column_name,
        logger,
        traditional_qsar_mode,
        model_type,
        feature_name_list,
        ignore_columns
    )
    #####################
    # 实例化Xenon对象 #
    #####################
    use_BOHB = env_utils.USE_BOHB
    if use_BOHB == "auto":
        use_BOHB = data.shape[0] > 10000
    imbalance_threshold = env_utils.IMBALANCE_THRESHOLD
    random_state = env_utils.RANDOM_STATE
    opt_framework = env_utils.OPT_FRAMEWORK
    total_time_limit = env_utils.TOTAL_TIME_LIMIT
    opt_early_stop_rounds = env_utils.OPT_EARLY_STOP_ROUNDS
    n_iterations = env_utils.N_ITERATIONS
    if random_state is None:
        logger.info("RANDOM_STATE is None, will choose a random_state between 0 and 10000.")
        random_state = np.random.randint(0, 10000)
    logger.info(f"random_state = {random_state}")
    kwargs = {
        "tuner": tuner,
        "hdl_constructor": hdl_constructor,
        "resource_manager": resource_manager,
        "log_path": f"{savedpath}/xenon.log",
        "random_state": random_state,
        "imbalance_threshold": imbalance_threshold,
        "use_xenon_opt": opt_framework == "xenon_opt",
        "total_time_limit": total_time_limit,
        "opt_early_stop_rounds": opt_early_stop_rounds,
        "n_iterations": n_iterations,
    }
    if use_BOHB:
        kwargs["use_BOHB"] = True
    if model_type == "clf":
        xenon = XenonClassifier(**kwargs)
    else:
        xenon = XenonRegressor(**kwargs)
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
    # multi-class clf metrics
    from xenon.metrics import roc_auc_ovo_macro, roc_auc_ovo_weighted, roc_auc_ovr_macro, roc_auc_ovr_weighted, \
        f1_macro, f1_micro, f1_weighted
    #  { 'roc_auc_ovo_macro': 1.0,
    # 'roc_auc_ovo_weighted': 1.0,
    # 'roc_auc_ovr_macro': 1.0,
    # 'roc_auc_ovr_weighted': 1.0,
    # 'precision_macro': 1.0,
    # 'precision_micro': 1.0,
    # 'precision_weighted': 1.0,
    # 'recall_macro': 1.0,
    # 'recall_micro': 1.0,
    # 'recall_weighted': 1.0,
    # 'f1_macro': 1.0,
    # 'f1_micro': 1.0,
    # 'f1_weighted': 1.0}

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
        # multi-class clf
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'roc_auc_ovo_macro': roc_auc_ovo_macro,
        'roc_auc_ovo_weighted': roc_auc_ovo_weighted,
        'roc_auc_ovr_macro': roc_auc_ovr_macro,
        'roc_auc_ovr_weighted': roc_auc_ovr_weighted,
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
    if SPLIT is not None:
        logger.info("User specific SPLIT, using SPLIT instead of KFOLD")
        logger.info("indicator 'TRAIN' is the train-set samples, other are validation-set samples.")
        mask = (SPLIT == "VALID")
        valid_indices = np.arange(data.shape[0])[mask]
        n_valids = valid_indices.size
        logger.info(f"{n_valids} train-set samples, {data.shape[0] - n_valids} valid-set samples")
        splitter = PredefinedHoldoutSplit(valid_indices)
    else:
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
            if model_type == "clf":
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=kfold, random_state=0)
            else:
                splitter = ShuffleSplit(n_splits=1, test_size=kfold, random_state=0)
        else:
            raise ValueError(f"Invalid KFOLD {kfold}.")
        logger.info(f"KFOLD = {kfold}, data_splitter will be interpret as {splitter}.")
    # ------------------------#
    # stacking 集成学习的参数   #
    # ------------------------#
    ensemble_size = env_utils.ENSEMBLE_SIZE
    if ensemble_size <= 1:
        logger.info("Xenon will not do stacking.")
        fit_ensemble_params = False
    else:
        logger.info(f"Xenon will stack {ensemble_size} models.")
        fit_ensemble_params = {"trials_fetcher_params": {"k": ensemble_size}}
    # --------------#
    # 启动Xenon     #
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
    # 满足集成学习可视化的需求
    if hasattr(xenon, "ensemble_estimator") and isinstance(xenon.ensemble_estimator, StackEstimator):
        display(resource_manager, xenon.task_id, 100, savedpath, trial_ids=xenon.trial_ids,  # 感觉输出csv的代码有点问题，就不输出了
                ensemble_estimator=xenon.ensemble_estimator, file_name="ensemble_records", output_csv=False,
                xenon=xenon)
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
                env_utils.DISPLAY_SIZE, savedpath, xenon=xenon)
    return xenon


if __name__ == '__main__':
    search(datapath=None, save_in_savedpath=True)
