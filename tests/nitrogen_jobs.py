#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from xenon.tests.base import SimulateNitrogenTestCase
from sklearn.metrics import accuracy_score
import json


class TestClfEnsembleSingle(SimulateNitrogenTestCase):
    TRIAL_ID = [12041]
    env_str = f"PYTHONUNBUFFERED=1;USER_ID=2;" \
              f"USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;" \
              f"TASK_ID=a581d55e299de5f52cbfc3bf5c643ee7;TRIAL_ID={TRIAL_ID};" \
              f"EXTERNAL_DELIVERY=True"
    name = "test_clf_ensemble_single"
    script = "ensemble"

    def do_test(self, savedpath: Path, datapath: Optional[Path] = None):
        '''查看文件是否存在'''
        assert (savedpath / f"feature_importance.csv").exists()
        assert (savedpath / f"selected_columns.json").exists()


class TestClfEnsembleStacking(SimulateNitrogenTestCase):
    TRIAL_ID = [12041, 12047, 12038]
    env_str = f"PYTHONUNBUFFERED=1;USER_ID=2;" \
              f"USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;" \
              f"TASK_ID=a581d55e299de5f52cbfc3bf5c643ee7;TRIAL_ID={TRIAL_ID};" \
              f"EXTERNAL_DELIVERY=True"
    name = "test_clf_ensemble_stacking"
    script = "ensemble"

    def do_test(self, savedpath: Path, datapath: Optional[Path] = None):
        '''查看文件是否存在'''
        for item in self.TRIAL_ID + ["stacking"]:
            assert (savedpath / f"feature_importance_{item}.csv").exists()
            if item != "stacking":
                assert (savedpath / f"selected_columns_{item}.json").exists()


class TestClfPredictFromSingle(SimulateNitrogenTestCase):
    EXPERIMENT_ID = 679
    dataset_id = 103252
    dataset_name = "job_33483_result"
    download_nitrogen_data = True
    env_str = f"PYTHONUNBUFFERED=1;USER_ID=2;" \
              f"USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;" \
              f"EXPERIMENT_ID={EXPERIMENT_ID};FEATURE_NAME_LIST=['AP2D'];" \
              f"EXTERNAL_DELIVERY=True"
    name = "test_clf_predict_single"
    script = "predict"

    def do_test(self, savedpath: Path, datapath: Optional[Path] = None):
        prediction_file = savedpath / f"prediction.csv"
        source_data_file = datapath / "data" / "data.csv"
        assert prediction_file.exists()
        prediction = pd.read_csv(prediction_file)
        source_data = pd.read_csv(source_data_file)
        for column in ("ID", "RESULT", "PROBA_0", "PROBA_1"):
            assert column in prediction.columns
        assert np.all((prediction["PROBA_1"] > 0.5).astype(int) == prediction["RESULT"])
        acc = accuracy_score(source_data["active"], prediction["RESULT"])
        print(acc)
        assert acc > 0.7


class TestClfPredictFromStacking(SimulateNitrogenTestCase):
    EXPERIMENT_ID = 680
    dataset_id = 103252
    dataset_name = "job_33483_result"
    download_nitrogen_data = True
    env_str = f"PYTHONUNBUFFERED=1;USER_ID=2;" \
              f"USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;" \
              f"EXPERIMENT_ID={EXPERIMENT_ID};FEATURE_NAME_LIST=['AP2D'];" \
              f"EXTERNAL_DELIVERY=True"
    name = "test_clf_predict_stacking"
    script = "predict"

    def do_test(self, savedpath: Path, datapath: Optional[Path] = None):
        prediction_file = savedpath / f"prediction.csv"
        source_data_file = datapath / "data" / "data.csv"
        assert prediction_file.exists()
        prediction = pd.read_csv(prediction_file)
        source_data = pd.read_csv(source_data_file)
        for column in ("ID", "RESULT", "PROBA_0", "PROBA_1"):
            assert column in prediction.columns
        assert np.all((prediction["PROBA_1"] > 0.5).astype(int) == prediction["RESULT"])
        acc = accuracy_score(source_data["active"], prediction["RESULT"])
        print(acc)
        assert acc > 0.7
        # 适配ensemble部分
        ensemble_info_file = savedpath / "ensemble_info.csv"
        assert ensemble_info_file.exists()
        trial_ids = pd.read_csv(ensemble_info_file)["trial_id"].tolist()
        trial_ids.remove("stacking")
        for trial_id in trial_ids:
            assert (savedpath / f"prediction_{trial_id}.csv").exists()
