#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-09-09
# @Contact    : qichun.tang@bupt.edu.cn
# clf metrics
from xenon.metrics import accuracy, mcc, sensitivity, specificity, \
    balanced_accuracy, f1, precision, recall, pac_score, auc, Scorer
# multi-class clf metrics
from xenon.metrics import roc_auc_ovo_macro, roc_auc_ovo_weighted, roc_auc_ovr_macro, roc_auc_ovr_weighted, \
    f1_macro, f1_micro, f1_weighted

# reg metrics
from xenon.metrics import r2, mean_squared_error, median_absolute_error

metric_dict = {
    # clf
    "accuracy": accuracy,
    "mcc": mcc,
    "auc": auc,
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