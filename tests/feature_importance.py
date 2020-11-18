#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

from xenon.tests.base import SimulateNitrogenTestCase


class TestFeatureImportance_Clf_Single(SimulateNitrogenTestCase):
    TRIAL_ID = [12041]
    env_str = f"PYTHONUNBUFFERED=1;USER_ID=2;SAVEDPATH=savedpath;" \
              f"USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;" \
              f"TASK_ID=a581d55e299de5f52cbfc3bf5c643ee7;TRIAL_ID={TRIAL_ID}"
    name = "test_feat_imp_clf_single"
    script = "ensemble"

    def do_test(self, savedpath: Path):
        assert (savedpath / f"feature_importance.csv").exists()
        assert (savedpath / f"selected_columns.json").exists()


class TestFeatureImportance_Clf_Stacking(SimulateNitrogenTestCase):
    TRIAL_ID = [12041, 12047, 12038]
    env_str = f"PYTHONUNBUFFERED=1;USER_ID=2;SAVEDPATH=savedpath;" \
              f"USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;" \
              f"TASK_ID=a581d55e299de5f52cbfc3bf5c643ee7;TRIAL_ID={TRIAL_ID}"
    name = "test_feat_imp_clf_stacking"
    script = "ensemble"

    def do_test(self, savedpath: Path):
        for item in self.TRIAL_ID + ["stacking"]:
            assert (savedpath / f"feature_importance_{item}.csv").exists()
            if item != "stacking":
                assert (savedpath / f"selected_columns_{item}.json").exists()
