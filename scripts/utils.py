#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import ast
import os
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import List, Optional
from typing import Tuple

import joblib
import json5 as json
import numpy as np
import pandas as pd
from tabulate import tabulate
from joblib import load
from scripts import lib_display
from xenon.interpret.feat_imp import get_feature_importances_in_xenon
from xenon.tools.external_delivery import transform_xenon
from xenon.utils.data import read_csv
from xenon.utils.logging_ import get_logger

util_logger = get_logger(__name__)
from xenon.utils.logging_ import get_logger


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
            try:
                return json.loads(value)
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

    def print(self, logger=None):
        if logger is None:
            func = print
        else:
            func = logger.info
        func("\n" + str(self))
        if len(self.long_data) > 0:
            func("--------------------")
            func("| Complex variable |")
            func("--------------------")

            for k, v in self.long_data:
                func(k + " : " + type(v).__name__)
                func(v)
                func("-" * 50)
        env_s = ";".join(
            [f"{k}={v}" for k, v in self.variables.items() if v not in ["None", None] and not k.endswith("WORKFLOW")] +
            ["SAVEDPATH=savedpath"]
        )
        func("env string for debug in pycharm:")
        func(env_s)

    __repr__ = __str__


def load_data_from_datapath(
        datapath,
        train_target_column_name,
        id_column_name,
        logger,
        traditional_qsar_mode,
        model_type,
        feature_name_list,
        ignore_columns
) -> Tuple[pd.DataFrame, dict, Optional[pd.Series], Optional[pd.Series]]:
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
        data = read_csv(f"{datapath}/data.csv")
        SMILES = None
        # if "SMILES" in data: # 理解错了
        #     SMILES = data.pop("SMILES")
        other_id_column = None
        if id_column_name is not None and id_column_name != "NAME" and id_column_name in data.columns:
            other_id_column = data.pop(id_column_name)
        # 添加主键
        for name_col_name in ("Name", "NAME", "ID", None):
            assert name_col_name is not None, ValueError("Name col name no found!")
            if name_col_name in data.columns:
                break
        needed_columns = [name_col_name]
        # 添加目标列
        if train_target_column_name is not None:
            logger.info(f"MODEL_TYPE = {model_type}")
            logger.info(f"TRAIN_TARGET_COLUMN_NAME = {train_target_column_name}")
            assert train_target_column_name in data.columns, ValueError(
                f"TRAIN_TARGET_COLUMN_NAME {train_target_column_name} do not exist in data.csv")
            needed_columns.append(train_target_column_name)
        data = data[needed_columns]
        feature_dir = f"{datapath}/feature"
        feature_file_list = []
        for feature_file in Path(feature_dir).iterdir():
            FP_name = feature_file.name.split(".")[0]
            if feature_name_list is not None and FP_name not in feature_name_list:
                logger.info(f"{FP_name} is ignored.")
                continue
            feature_file_list.append(feature_file.as_posix())
        for feature_file in sorted(feature_file_list):
            df = read_csv(feature_file)
            data = data.merge(df, on=name_col_name)
        if other_id_column is not None:
            data[id_column_name] = other_id_column
            data.pop(name_col_name)
            name_col_name = id_column_name
        column_descriptions = {
            "target": train_target_column_name,
            "id": name_col_name
        }
    else:
        column_descriptions = {}
        column_descriptions["target"] = train_target_column_name
        data = read_csv(datapath)
        SMILES = None
        # if "SMILES" in data: # 理解错了
        #     SMILES = data.pop("SMILES")
        if "target" not in column_descriptions:
            column_descriptions["target"] = train_target_column_name
        if id_column_name is not None and id_column_name in data.columns:
            column_descriptions["id"] = id_column_name
        if train_target_column_name is not None:
            assert train_target_column_name in data.columns, ValueError(
                f"TRAIN_TARGET_COLUMN_NAME {train_target_column_name} do not exist in data.csv")
    column_descriptions['ignore'] = ignore_columns
    SPLIT = None
    if "SPLIT" in data.columns:
        SPLIT = data.pop("SPLIT")
        logger.info("Train Data using custom data split method.")
        logger.info(f"SPLIT statistics: {Counter(SPLIT)}")
    logger.info(f"Data Loading successfully. Data Shape: {data.shape}")
    return data, column_descriptions, SPLIT, SMILES


def save_current_expriment_model(savedpath, experiment_id, logger, xenon):
    final_model_path = f"{savedpath}/experiment_{experiment_id}_best_model.bz2"
    logger.info(
        f"Experiment(experiment_id={experiment_id}) is finished, the best model will be saved in {final_model_path}.")
    final_model = xenon.copy()
    # prevent pickle error in tiny enviroment
    final_model.shps = None
    final_model.tuner.shps = None
    final_model.tuners = None
    final_model.hdl_constructors = None
    joblib.dump(final_model, final_model_path)
    logger.info("save feature importance to SAVEDPATH")
    outputs = get_feature_importances_in_xenon(xenon)
    # 一个列表, 每个item是stacking模型每个base-model的特征重要度
    feature_importances: List[pd.DataFrame] = [output[0] for output in outputs]
    #                                            的特征筛选列
    selected_columns_list: List[pd.Index] = [output[1] for output in outputs]
    if len(feature_importances) == 1:
        feature_importances[0].to_csv(f"{savedpath}/feature_importance.csv")
        json.dump(selected_columns_list[0].tolist(), open(f"{savedpath}/selected_columns.json", "w+"))
    else:
        N = len(feature_importances)
        if hasattr(xenon, "trial_ids"):
            assert len(xenon.trial_ids) == N, ValueError
            suffixs = [f"{trial_id}" for trial_id in xenon.trial_ids]
            weights = np.array(xenon.weights)
            weights /= weights.sum()
        else:
            suffixs = [f"{i}" for i in range(N)]
            weights = np.ones([N]) / N
        s_df = feature_importances[0]
        final_feature_importance = pd.DataFrame(np.zeros(s_df.shape), columns=s_df.columns, index=s_df.index)
        for suffix, feature_importance, selected_columns, weight in \
                zip(suffixs, feature_importances, selected_columns_list, weights):
            feature_importance.to_csv(f"{savedpath}/feature_importance_{suffix}.csv")
            json.dump(selected_columns.tolist(), open(f"{savedpath}/selected_columns_{suffix}.json", "w+"))
            final_feature_importance += weight * feature_importance
        final_feature_importance.to_csv(f"{savedpath}/feature_importance_stacking.csv")
    # 产生对外交付模型
    if os.getenv("EXTERNAL_DELIVERY", "").lower() == "true":
        print("start external_delivery")
        external_delivery(xenon, savedpath, logger)


def display(
        resource_manager, task_id, display_size, savedpath,
        trial_ids=None, ensemble_estimator=None, file_name="search_records", output_csv=True
):
    user_id = resource_manager.user_id
    if trial_ids is None:
        records = resource_manager._get_sorted_trial_records(task_id, user_id, display_size)
    else:
        records = resource_manager._get_trial_records_by_ids(trial_ids, task_id, user_id)
    ml_task, y_train = resource_manager.get_ensemble_needed_info(task_id)
    y_train = y_train.data
    # 处理records, 加载y_info_path
    processed_records = []
    records_copy = deepcopy(records)
    for record in records_copy:
        y_info_path = record["y_info_path"]
        trial_id = record["trial_id"]
        exception = None
        # keys: ['y_true_indexes', 'y_preds', 'y_test_pred']
        try:
            y_info = resource_manager.file_system.load_pickle(y_info_path)
            record["y_info"] = y_info
            processed_records.append(record)
        except Exception as e:
            exception = str(e)
            parser_logger.error(f"error trial_id = {trial_id}")
        if exception is not None:
            parser_logger.error(exception)
    # 处理stacking ensemble的可视化
    # fixme: 但是输出csv有点问题，把下面的records改成processed_records会好一点
    if ensemble_estimator is not None:
        ensemble_record = processed_records[0].copy()
        for k, v in ensemble_record.items():
            if isinstance(v, (int, float, bool)):
                ensemble_record[k] = 0
            else:
                ensemble_record[k] = ''
        all_score = ensemble_estimator.all_score
        ensemble_record['all_score'] = ensemble_estimator.all_score
        if ensemble_estimator.confusion_matrix:
            ensemble_record['additional_info'] = {"confusion_matrices": [ensemble_estimator.confusion_matrix]}
        ensemble_record['estimator'] = 'stacking'  # fixme: 浮点精度截断
        ensemble_record['estimating'] = str(
            dict(zip(trial_ids, [float(f"{w:.3f}") for w in ensemble_estimator.weights])))
        if 'mcc' in all_score:
            ensemble_record['loss'] = 1 - all_score['mcc']
        else:
            ensemble_record['loss'] = 1 - all_score['r2']  # fixme: 应该没写错吧
        y_true_indexes = processed_records[0]['y_info']['y_true_indexes']
        # hold-out
        if len(y_true_indexes) == 1:
            y_preds = [ensemble_estimator.stacked_y_pred]
        # 交叉验证
        else:
            y_preds = []
            for ix in y_true_indexes:
                y_preds.append(ensemble_estimator.stacked_y_pred[ix])
        y_info = {"y_true_indexes": y_true_indexes, "y_preds": y_preds}
        ensemble_record["y_info"] = y_info
        ensemble_record["trial_id"] = "stacking"
        processed_records.append(ensemble_record)
    # 用于输出xenon每折的预测结果
    if len(processed_records) > 0:
        hstack = np.hstack(processed_records[0]["y_info"]["y_true_indexes"])
        pred_df = pd.DataFrame(index=list(range(hstack.max() + 1)))
        # 产品要求输出主键名
        experiment_id = os.getenv("EXPERIMENT_ID")
        if experiment_id:
            experiment_records = resource_manager._get_experiment_record(experiment_id)
            assert len(experiment_records) > 0, ValueError(f"experiment_id {experiment_id} is invalid.")
            experiment_record = experiment_records[0]
            task_id = experiment_record.get("task_id", experiment_record.get("task"))
            final_model_path = experiment_record["final_model_path"]
            local_path = f"{savedpath}/experiment_{experiment_id}_best_model.bz2"
            # 下载过了就不用了，直接load
            if not os.path.exists(local_path):
                # 判断非空
                assert bool(final_model_path), ValueError(
                    f"experiment {experiment_id}  was not completed normally, is invalid.")
                resource_manager.file_system.download(final_model_path, local_path)
            xenon = load(local_path)
            train_id_seq = getattr(xenon.data_manager, "train_id_seq", None)
            ID_COLUMN_NAME = xenon.data_manager.column_descriptions.get('id')
            if train_id_seq is not None and ID_COLUMN_NAME is not None:
                pred_df[ID_COLUMN_NAME] = train_id_seq
        # 输出完毕
        for record in processed_records:
            trial_id = record["trial_id"]
            y_true_indexes = record["y_info"]["y_true_indexes"]
            y_preds = record["y_info"]['y_preds']
            n_folds = len(y_preds)
            fold_col = f"{trial_id}_FOLD"
            for fold_id in range(n_folds):
                y_true_index = y_true_indexes[fold_id]
                y_pred = y_preds[fold_id]
                if ml_task.mainTask == "classification":
                    for proba_id in range(y_pred.shape[1]):
                        pred_df.loc[y_true_index, f"{trial_id}_PROBA_{proba_id}"] = y_pred[:, proba_id]
                else:
                    pred_df.loc[y_true_index, f"{trial_id}"] = y_pred
                pred_df.loc[y_true_index, fold_col] = fold_id
            # pred_df[fold_col] = pred_df[fold_col].astype('int32')
        pred_df.to_csv(f"{savedpath}/{file_name}_predictions.csv", index=False)
    # 结束输出xenon每折的预测结果
    data = {
        "mainTask": ml_task.mainTask,
        "records": processed_records,
        "y_train": y_train
    }

    output_records = deepcopy(records)
    for output_record in output_records:
        output_record['budget'] = output_record['additional_info'].get('budget')
        output_record.pop("all_scores")
        output_record.pop("intermediate_results")
        output_record.pop("test_all_score")
        output_record.pop("losses")
        output_record.update(output_record.pop("all_score"))
    search_records_df = pd.DataFrame(output_records)
    search_records_csv_path = f"{savedpath}/{file_name}.csv"
    search_records_html_path = f"{savedpath}/{file_name}.html"
    if output_csv:
        search_records_df.to_csv(search_records_csv_path, index=False)
    try:
        Path(search_records_html_path).write_text(lib_display.display(data))
    except Exception as e:
        util_logger.error(e)


def save_info_json(experiment_id, task_id, hdl_id, savedpath):
    info = {}
    if experiment_id is not None:
        info["experiment_id"] = experiment_id
    if task_id is not None:
        info["task_id"] = task_id
    if hdl_id is not None:
        info["hdl_id"] = hdl_id
    Path(f"{savedpath}/info.json").write_text(json.dumps(info))


parser_logger = get_logger("Previous_Result_Parser")


def is_xenon_previous_result_dataset(datapath):
    name = Path(datapath).name
    if name.endswith("result") and Path(datapath).is_dir():
        parser_logger.info(f"{name} endswith 'result' , maybe is previous Xenon result dataset.")
        if (Path(datapath) / "info.json").is_file():
            parser_logger.info(f"'info.json' exists in {name}, I'm sure it's previous Xenon result dataset.")
            return True
        else:
            parser_logger.info(f"'info.json' not exists in {name},  it's not previous Xenon result dataset!!!")
            return False
    return False


def set_xenon_several_IDs_to_env_by_datapath(datapath):
    info = json.loads((Path(datapath) / "info.json").read_text())
    parser_logger.info(f"parsing 'info.json' in '{datapath}' ...")
    for k, v in info.items():
        k = k.upper()
        parser_logger.info(f"set {k}\t=\t'{v}'\tOK")
        os.environ[k] = str(v)


def process_previous_result_dataset():
    # nitrogen 本地的 result_dataset 为 job_xxx_result
    # xbcp           result_dataset 为  xxx-result
    # nitrogen 多个dataset的DATAPATH /home/job/data
    datapath = os.getenv("DATAPATH")
    if bool(datapath) and isinstance(datapath, str) and os.path.isdir(datapath):
        parser_logger.info(f"DATAPATH {datapath} exists, and is dir")
        # 情况1 DATAPATH 是一个单独的result dataset, 如 DATAPATH = /home/qichuntang/xxxx_result
        if is_xenon_previous_result_dataset(datapath):
            set_xenon_several_IDs_to_env_by_datapath(datapath)
        else:
            # 情况2 DATAPATH 是 result_dataset + data_input, 如 DATAPATH = /home/qichuntang/xxxx_result
            sub_dirs = os.listdir(datapath)
            xenon_results = []
            data_inputs = []
            for sub_dir in sub_dirs:
                path = Path(datapath) / sub_dir
                if is_xenon_previous_result_dataset(path):
                    xenon_results.append(path)
                else:
                    data_inputs.append(path)
            if len(xenon_results) == 0:
                parser_logger.info("current DATAPATH don't contain previous Xenon result dataset. ")
            else:
                parser_logger.info("=== Here is previous Xenon result dataset ===")
                for xenon_result in xenon_results:
                    parser_logger.info(f"* {xenon_result}")
                if len(xenon_results) > 1:
                    parser_logger.info(f"len(xenon_results) = {len(xenon_results)}, only use first!")
                xenon_result = xenon_results[0]
                set_xenon_several_IDs_to_env_by_datapath(xenon_result)
                if len(data_inputs) > 0:
                    parser_logger.info("=== Here is data input dir ===")
                    for data_input in data_inputs:
                        parser_logger.info(f"* {data_input}")
                    if len(data_inputs) > 1:
                        parser_logger.info(f"len(data_inputs) = {len(data_inputs)}, only use first!")
                    data_input = str(data_inputs[0])
                    parser_logger.info(f"set DATAPATH\t=\t'{data_input}'\tOK")
                    os.environ["DATAPATH"] = data_input


def print_xenon_path(logger=None):
    if logger is None:
        func = print
    else:
        func = logger.info
    import xenon
    func(f"Xenon executable file: {xenon.__file__}")


def external_delivery(xenon, savedpath=".", logger=None):
    if logger is None:
        func = print
    else:
        func = logger.info
    sk_model = transform_xenon(xenon)
    feature_names = xenon.feature_names
    root = Path(__file__).parent.parent
    subprocess.check_call([sys.executable, 'setup_ext.py', 'bdist_wheel'], cwd=root.as_posix())
    # tmp_path = f"/tmp/{uuid.uuid4()}"
    # os.system(f"mkdir {tmp_path}")
    dist_dir = f"{root}/dist"
    fname = os.listdir(dist_dir)[0]
    # os.system(f"mv {dist_dir}/{fname} {tmp_path}/")
    external_delivery_path = Path(savedpath) / "external_delivery"
    external_delivery_path.mkdir(parents=True, exist_ok=True)
    func(f"xenon_ext install bag: {external_delivery_path}/{fname}")
    os.system(f"mv {dist_dir}/{fname} {external_delivery_path}/{fname}")
    os.system(f"rm -rf  {root}/build {root}/dist {root}/*.egg-info")
    func(f"external_delivery model path: {external_delivery_path}/model.bz2")
    joblib.dump(sk_model, open(f"{external_delivery_path}/model.bz2", "wb+"))
    func(f"model's feature_names: {external_delivery_path}/feature_names.json")
    json.dump(list(feature_names), open(f"{external_delivery_path}/feature_names.json", "w+"))
    func(f"mock_data csv for unittest: {external_delivery_path}/mock_data.csv")
    mock_data = pd.DataFrame(np.random.rand(10, len(feature_names)), columns=feature_names)
    mock_data.to_csv(f"{external_delivery_path}/mock_data.csv", index=False)
    func(f"test.py: {external_delivery_path}/test.py")
    os.system(f"cp {root}/xenon_ext/test.py {external_delivery_path}/test.py")
    func(f"Makefile: {external_delivery_path}/Makefile")
    os.system(f"cp {root}/xenon_ext/Makefile {external_delivery_path}/Makefile")
    os.system(f"cd {savedpath} && tar -zcvf external_delivery.tar.gz external_delivery --remove-files ")


if __name__ == '__main__':
    env = EnvUtils()
    env.from_json('env_configs/search.json')
    print(env)
    print(env)
    # xenon = load("/home/tqc/Project/Xenon/savedpath/test_feat_imp_clf_stacking_2/experiment_680_best_model.bz2")
    # external_delivery(xenon, "/home/tqc/Project/Xenon/savedpath/ext")
