#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-23
# @Contact    : qichun.tang@bupt.edu.cn
from xenon_ext.occ_model.gbdt_lr import GBDT_LR_Classifier, GBDT_LR_Regressor
from xenon_ext.occ_model.occ_lgbm import OCC_LGBMClassifier, OCC_LGBMRegressor, OCC_RFClassifier, OCC_RFRegressor
from xenon_ext.occ_model.xlearn import XlearnClassifier, XlearnRegressor

mapper = {
    "lgbm_gbdt": {
        "classification": OCC_LGBMClassifier,
        "regression": OCC_LGBMRegressor,
    },
    "lgbm_dart": {
        "classification": OCC_LGBMClassifier,
        "regression": OCC_LGBMRegressor,
    },
    "lgbm_gbdt_lr_l1": {
        "classification": GBDT_LR_Classifier,
        "regression": GBDT_LR_Regressor,
    },
    "lgbm_gbdt_lr_l2": {
        "classification": GBDT_LR_Classifier,
        "regression": GBDT_LR_Regressor,
    },
    "lgbm_rf": {
        "classification": OCC_RFClassifier,
        "regression": OCC_RFRegressor,
    },
    "xl_lr_adagrad": {
        "classification": XlearnClassifier,
        "regression": XlearnRegressor,
    },
    "xl_fm_adagrad": {
        "classification": XlearnClassifier,
        "regression": XlearnRegressor,
    },
    "xl_lr_ftrl": {
        "classification": XlearnClassifier,
        "regression": XlearnRegressor,
    },
    "xl_fm_ftrl": {
        "classification": XlearnClassifier,
        "regression": XlearnRegressor,
    },
}
