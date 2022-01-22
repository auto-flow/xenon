#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-07
# @Contact    : qichun.tang@bupt.edu.cn
###################################
# Insert current path to sys path #
###################################
import os
import sys

sys.path.insert(0, os.getcwd())
# fixme: 用于解决xlearn在xenon镜像中的问题
# https://github.com/aksnzhy/xlearn/issues/215
os.environ["USER"] = "test"
###################################
import json
import logging
import threading
from pathlib import Path
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from mlxtend.evaluate import PredefinedHoldoutSplit
from sklearn.base import TransformerMixin
from sklearn.datasets import dump_svmlight_file
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, StratifiedShuffleSplit, ShuffleSplit

from scripts.utils import EnvUtils, load_data_from_datapath, print_xenon_path, parse_env_estimator_params
from xenon.utils.logging_ import setup_logger
from xenon4bigdata.feature_selection import L1Linear_FeatureSelection, DummyFeatureSelection, GBDT_FeatureSelection, \
    RandomForest_FeatureSelection
from xenon_ext.selector import Xenon4bigdata_FeatureSelector

os.system(f'mkdir -p $SAVEDPATH')
env_utils = EnvUtils()
env_utils.from_json("env_configs/common.json")
env_utils.from_json("env_configs/bd1_preprocess.json")
env_utils.update()
logger = logging.getLogger("bd1_preprocess.py")
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
traditional_qsar_mode = True
if os.path.isdir(datapath):
    logger.info("DATAPATH 为传统QSAR模式，传入的是分子指纹矢量化后的结果。")
else:
    traditional_qsar_mode = False
    logger.info("DATAPATH 为用户自定义模式，传入的是用户自定义的特征文件。")
#     logger.info("""需要注意的是，该模式下用户需要指定'COLUMN_DESCRIPTIONS'环境变量
# 如：
# COLUMN_DESCRIPTIONS = {'id' : "NAME" ,'target' : 'pIC50','ignore' : ['SMILES']}
# """)
logger.info(f"traditional_qsar_mode = {traditional_qsar_mode}")
env_utils.print(logger)
######################
# 从DATAPATH中加载数据 #
######################
feature_name_list = env_utils.FEATURE_NAME_LIST
column_descriptions = env_utils.COLUMN_DESCRIPTIONS
train_target_column_name = env_utils.TRAIN_TARGET_COLUMN_NAME
id_column_name = env_utils.ID_COLUMN_NAME
model_type = env_utils.MODEL_TYPE
selector_params = parse_env_estimator_params(env_utils.SELECTOR_PARAMS)
logger.info(f"selector_params = {selector_params}")
# 公用的数据加载部分（SPLIT表示自定义切分）
start_time = time()
logger.info('开始加载数据... ')
fp_to_columns = {}
data, column_descriptions, SPLIT, _ = load_data_from_datapath(
    datapath,
    train_target_column_name,
    id_column_name,
    logger,
    traditional_qsar_mode,
    model_type,
    feature_name_list,
    column_descriptions,
    False,
    fp_to_columns,
    IGNORE_COLUMNS=env_utils.IGNORE_COLUMNS
)
id_col = None
if 'id' in column_descriptions:
    id_column_name = column_descriptions['id']
    id_col = data.pop(id_column_name)
cost_time = time() - start_time
logger.info(f'done, 耗时【{cost_time:.2f}】s')

y = data.pop(train_target_column_name).values
columns = list(data.columns)
# assert train_target_column_name in columns, ValueError('你设置的TRAIN_TARGET_COLUMN_NAME不在数据中')
X = data
# 开始做特征筛选
logger.info(f'开始做特征筛选')
feature_select_method = env_utils.FEATURE_SELECT_METHOD
assert feature_select_method in [
    'none',
    'l1_linear',
    'gbdt',
    'rf',
]
choosed_selector_params = selector_params[feature_select_method]
logger.info(f"choosed_selector_params = {choosed_selector_params}")
# 设计模型：简单工厂模式
if feature_select_method == 'none':
    selector = DummyFeatureSelection()
elif feature_select_method == 'l1_linear':
    selector = L1Linear_FeatureSelection(choosed_selector_params)
elif feature_select_method == 'gbdt':
    selector = GBDT_FeatureSelection(choosed_selector_params)
elif feature_select_method == 'rf':
    selector = RandomForest_FeatureSelection(choosed_selector_params)
else:
    raise NotImplementedError()
logger.info(f'开始做特征筛选... X.shape = {X.shape}')
start_time = time()
X_selected, columns_selected = selector.select(X, y, columns, model_type)
cost_time = time() - start_time
n_del = X.shape[1] - X_selected.shape[1]
del_ratio = n_del / X.shape[1]
logger.info(f'done, 耗时【{cost_time:.2f}】s, \n'
            f'从【{X.shape[1]}】个特征中删除了【{n_del}】个特征，\n'
            f'删除后保留有【{X_selected.shape[1]}】个特征，\n'
            f'删除率【{del_ratio * 100:.2f}%】')

logger.info(f'开始构造和保存preprocessing阶段的Transformers，会在search阶段被上传到nitrogen和数据库。ensemble的时候会下载Transformers用于拼接Pipeline')
# 当前的Transformers只有一个简单的【特征筛选器】，TODO: 更丰富的preprocess
# 如果不做特征筛选， columns_selected 是全集columns，不影响【特征筛选器】的构造
# transformers 的数据结构就是 sklearn.pipeline.Pipeline 的steps参数要求的数据结构
transformers: List[Tuple[str, TransformerMixin]] = []
transformers.append(
    ('feature_selector', Xenon4bigdata_FeatureSelector(columns_selected))
)
dump(transformers, f"{savedpath}/transformers.pkl")

start_time = time()
# 统计分子指纹筛选情况
if fp_to_columns:  # 只有传统qsar模式才会激活这个option
    logger.info('统计分子指纹筛选情况')
    columns_selected_set = set(columns_selected)
    fp_fs_stats = []
    all_n_feats = 0
    all_n_selected = 0
    for fp, fp_columns in fp_to_columns.items():
        fp_columns_set = set(fp_columns)
        n_feats = len(fp_columns_set)
        n_selected = len(fp_columns_set & columns_selected_set)
        keep_ratio = n_selected / n_feats
        fp_fs_stats.append([fp, n_feats, n_selected, keep_ratio])
        all_n_feats += n_feats
        all_n_selected += n_selected
    all_keep_ratio = all_n_selected / all_n_feats
    fp_fs_stats.append(['sum', all_n_feats, all_n_selected, all_keep_ratio])
    fp_fs_stats_df = pd.DataFrame(fp_fs_stats, columns=['fp', 'n_feats', 'n_selected', 'keep_ratio'])
    fp_fs_stats_df.to_csv(savedpath + "/fp_fs_stats.csv", index=False)
    with open(savedpath + "/fp_to_columns.json", "w+") as f:
        json.dump(fp_to_columns, f)

USE_THREADING = False
THREADS: List[threading.Thread] = []


# 存储新的数据
def save_libsvm(X_selected, y, columns_selected, path, use_threading=USE_THREADING):
    '''存储lgbm可以识别的csv'''
    if not use_threading:
        # zero_based 必须为 True， 否则训练的lgbm有问题
        dump_svmlight_file(X_selected, y, path, zero_based=True, multilabel=False)
    else:
        t = threading.Thread(target=dump_svmlight_file, args=[X_selected, y, path],
                             kwargs=dict(zero_based=True, multilabel=False))
        t.start()
        THREADS.append(t)


# 存储总的数据
trainset_path = savedpath + '/train_full.txt'
# args_list = [[X_selected, y, columns_selected, trainset_path]]
save_libsvm(X_selected, y, columns_selected, trainset_path)

# 特征筛选后，删除筛选前的X腾出内存
# del X # 因为开线程存，就不删了

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

# 存储各个交叉验证的数据
# todo: 开10个线程做这件事
valid_set_indexes = []
for ix, (train_ix, valid_ix) in enumerate(splitter.split(X_selected, y)):
    valid_set_indexes.append(valid_ix)
    trainset_path = savedpath + f'/train_{ix}.txt'
    validset_path = savedpath + f'/valid_{ix}.txt'
    X_train = X_selected.loc[train_ix, :]
    y_train = y[train_ix]
    X_valid = X_selected.loc[valid_ix, :]
    y_valid = y[valid_ix]
    # args_list.append([X_train, y_train, columns_selected, trainset_path])
    # args_list.append([X_valid, y_valid, columns_selected, validset_path])
    save_libsvm(X_train, y_train, columns_selected, trainset_path)
    save_libsvm(X_valid, y_valid, columns_selected, validset_path)

dump(valid_set_indexes, savedpath + "/valid_set_indexes.pkl")

pd.DataFrame(y[:, None], columns=["LABEL"]).to_csv(savedpath + '/LABEL.csv', index=False)
if id_col is not None:
    pd.DataFrame(y[:, None], columns=[id_column_name]).to_csv(savedpath + '/NAME.csv', index=False)
feat_imp = pd.DataFrame(np.array(columns)[:, None], columns=['column_name'])
feat_imp['feature_importances'] = selector.coef
feat_imp.to_csv(savedpath + '/feature_importances.csv', index=False)
with open(savedpath + '/columns.txt', 'w') as f:
    f.write(",".join([str(x) for x in columns_selected]))
selector.dump_model(savedpath + '/fs_model.pkl')
for t in THREADS:
    t.join()
cost_time = time() - start_time
logger.info(f"数据存储耗时【{cost_time:.2f}s】")
logger.info('done.')
# 用多线程搞容易把内存扩大6倍？
# Parallel(backend='threading', n_jobs=11)( # 这个代码主要吃IO
#     delayed(save_libsvm)(*args) for args in args_list
# )
