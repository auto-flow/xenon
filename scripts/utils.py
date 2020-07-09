#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import ast
import json
import os
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Tuple

import joblib
import pandas as pd
from tabulate import tabulate


class EnvUtils:
    def __init__(self):
        self.env_items = []
        self.variables = {}

    def from_json(self, json_path):
        env_items = json.loads((Path(__file__).parent / json_path).read_text())
        for env_item in env_items:
            self.add_env(**env_item)

    def add_env(self, name, default, description):
        self.env_items.append({
            "name": name,
            "default": default,
            "description": description,
        })
        self.variables[name] = default

    def __getitem__(self, item):
        return self.variables[item]

    def __getattr__(self, item):
        return self.variables[item]

    def update(self):
        for item in self.env_items:
            name = item["name"]
            value = os.getenv(name)
            if value is not None:
                value = value.strip()
                parsed_value = self.parse(value)
                if parsed_value is not None:
                    self.variables[name] = parsed_value

    def parse(self, value: str):
        if value.lower() in ("null", "none", "nan"):
            return None
        try:
            return ast.literal_eval(value)
        except:
            return value

    def get_data(self):
        data = []
        long_data = []
        for k, v in self.variables.items():
            sv = str(v)
            if len(sv) < 20:
                data.append([k, sv])
            else:
                long_data.append([k, v])
        return data, long_data

    def __str__(self):
        data, self.long_data = self.get_data()
        return tabulate(data, headers=["name", "value"])

    def print(self):
        print(self)
        if len(self.long_data) > 0:
            print("--------------------")
            print("| Complex variable |")
            print("--------------------")

            for k, v in self.long_data:
                print(k + " : " + type(v).__name__)
                pprint(v)
                print("-" * 50)

    __repr__ = __str__


def load_data_from_datapath(
        datapath,
        train_target_column_name,
        logger,
        traditional_qsar_mode,
        model_type,
        feature_name_list,
        column_descriptions
) -> Tuple[pd.DataFrame, dict]:
    if traditional_qsar_mode:
        # 情况一 ： DATAPATH = job_xxx_result/
        # job_xxx_result/ (DATAPATH)
        #      data/
        #         data.csv
        #         feature/
        #            2D.csv
        #            ...
        # 情况二 ： DATAPATH = data/
        #      data/ (DATAPATH)
        #         data.csv
        #         feature/
        #            2D.csv
        #            ...
        # 注： 用户上传数据集时应避免情况二的出现，因为会出现文件夹重名的情况
        if os.path.isdir(f"{datapath}/data"):
            # 情况一 满足这个判断条件
            datapath = f"{datapath}/data"
        # 情况二不满足判断条件
        data = pd.read_csv(f"{datapath}/data.csv")
        needed_columns = ["NAME"]
        if train_target_column_name is not None:
            logger.info(f"MODEL_TYPE = {model_type}")
            logger.info(f"TRAIN_TARGET_COLUMN_NAME = {train_target_column_name}")
            assert train_target_column_name in data.columns, ValueError(
                f"TRAIN_TARGET_COLUMN_NAME {train_target_column_name} do not exist in data.csv")
            needed_columns.append(train_target_column_name)
        data = data[needed_columns]
        feature_dir = f"{datapath}/feature"
        for feature_file in Path(feature_dir).iterdir():
            FP_name = feature_file.name.split(".")[0]
            if feature_name_list is not None and FP_name not in feature_name_list:
                logger.info(f"{FP_name} is ignored.")
                continue
            df = pd.read_csv(feature_file)
            data = data.merge(df, on="NAME")
        column_descriptions = {
            "target": train_target_column_name,
            "id": "NAME"
        }
    else:
        if "target" not in column_descriptions:
            column_descriptions["target"] = train_target_column_name
        data = pd.read_csv(datapath)
        if train_target_column_name is not None:
            assert train_target_column_name in data.columns, ValueError(
                f"TRAIN_TARGET_COLUMN_NAME {train_target_column_name} do not exist in data.csv")
    return data, column_descriptions


def save_current_expriment_model(savedpath, experiment_id, logger, xenon):
    final_model_path = f"{savedpath}/experiment_{experiment_id}_best_model.bz2"
    logger.info(
        f"Experiment(experiment_id={experiment_id}) is finished, the best model will be saved in {final_model_path}.")
    final_model = xenon.copy()
    joblib.dump(final_model, final_model_path)

def display(resource_manager,task_id, display_size,savedpath):
    user_id = resource_manager.user_id
    records = resource_manager._get_sorted_trial_records(task_id, user_id, display_size)
    ml_task, y_train = resource_manager.get_ensemble_needed_info(task_id)
    y_train = y_train.data
    # 处理records, 加载y_info_path
    processed_records = []
    records_=deepcopy(records)
    for record in records_:
        y_info_path = record["y_info_path"]
        # keys: ['y_true_indexes', 'y_preds', 'y_test_pred']
        y_info = resource_manager.file_system.load_pickle(y_info_path)
        record["y_info"] = y_info
        processed_records.append(record)
    output_records = deepcopy(records)
    for output_record in output_records:
        output_record.pop("all_scores")
        output_record.pop("intermediate_results")
        output_record.pop("test_all_score")
        output_record.pop("losses")
        output_record.update(output_record.pop("all_score"))
    search_records_df = pd.DataFrame(output_records)
    search_records_path = f"{savedpath}/search_records.csv"
    search_records_df.to_csv(search_records_path, index=False)
