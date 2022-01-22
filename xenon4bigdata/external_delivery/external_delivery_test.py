#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-10-09
# @Contact    : qichun.tang@bupt.edu.cn
import glob
import sys
from pathlib import Path

import pandas as pd
from joblib import load

assert tuple(sys.version_info)[:2] >= (3, 6), EnvironmentError(
    'Xenon环境要求Python 3.6+'
)

if __name__ == '__main__':
    cwd = Path(__file__).parent
    pattern = "experiment_*_best_model.bz2"
    model_candidates = list(glob.glob(f"{cwd}/{pattern}"))
    assert len(model_candidates) >= 1, \
        ValueError(f'external_delivery_test.py所在目录内没有"{pattern}" !')
    model_path = model_candidates[0]
    print(f'加载xenon模型: {model_path}')
    model = load(model_path)
    df = pd.read_csv(f"{cwd}/external_delivery_mock_data.csv")
    try:
        # 是分类器就尝试做概率预估
        y_pred = model.predict_proba(df)[:,-1]
    except:
        y_pred = model.predict(df)
    print('测试预估成功：')
    print(y_pred)
    exit(0)
