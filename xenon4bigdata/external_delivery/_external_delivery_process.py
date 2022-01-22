#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-09
# @Contact    : qichun.tang@bupt.edu.cn
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import xenon4bigdata


def external_delivery(models, savedpath=os.getenv('SAVEDPATH')):
    # fixme: 之前的做法，废弃，因为ensemble的上游数据不是preprocess
    # columns = Path(os.environ['DATAPATH'] + "/columns.txt").read_text().split(",")
    # fixme: 这个方法需要model是一个Pipeline，且step[0]是一个特征选择器
    columns = set()
    for model in models:
        columns |= set(model.steps[0][1].columns)
    root = Path(xenon4bigdata.__file__).parent.parent
    os.system(f"rm -rf {root}/build {root}/dist {root}/*.egg-info ")
    script_root = f"{root}/xenon4bigdata/external_delivery"
    # 生成xenon_ext.whl文件
    subprocess.check_call([
        sys.executable, 'setup_ext.py', 'bdist_wheel'], cwd=root.as_posix())
    dist_dir = f"{root}/dist"
    fname = os.listdir(dist_dir)[0]
    os.system(f"mv {dist_dir}/{fname} {savedpath}")
    M = len(columns)
    mock_data = pd.DataFrame(
        np.random.rand(10, M),
        columns=columns
    )
    mock_data.to_csv(f"{savedpath}/external_delivery_mock_data.csv")
    cp_func = lambda file: os.system(f"cp {script_root}/{file} {savedpath}/{file}")
    for file in [
        'README.md',
        'external_delivery_test.py',
    ]:
        cp_func(file)
    os.system(f"rm -rf {root}/build {root}/dist {root}/*.egg-info ")


if __name__ == '__main__':
    root = '/data/Project/AutoML/Xenon_v2.0'
    p = f'{root}/savedpath/xenon4bigdata/bd2_search_gbdt_lr'
    os.environ['DATAPATH'] = \
        f'{root}/savedpath/xenon4bigdata/bd1_preprocess_clf'
    external_delivery(p)
