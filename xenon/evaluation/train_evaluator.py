import datetime
import sys
from collections import defaultdict
from contextlib import redirect_stderr
from io import StringIO
from time import time
from typing import Dict, Optional, List

import numpy as np
from xenon.lazy_import import Configuration

from xenon.constants import PHASE2, PHASE1, SERIES_CONNECT_LEADER_TOKEN, SERIES_CONNECT_SEPARATOR_TOKEN
from xenon.data_container import DataFrameContainer
from xenon.data_manager import DataManager
from xenon.ensemble.utils import vote_predicts, mean_predicts
from xenon.evaluation.base import BaseEvaluator
from xenon.hdl.shp2dhp import SHP2DHP
from xenon.metrics import Scorer, calculate_score, calculate_confusion_matrix
from xenon.resource_manager.base import ResourceManager
from xenon.utils.dict_ import group_dict_items_before_first_token
from xenon.utils.logging_ import get_logger
from xenon.utils.ml_task import MLTask
from xenon.utils.packages import get_class_object_in_pipeline_components
from xenon.utils.pipeline import concat_pipeline
from xenon.utils.sys_ import get_trance_back_msg
from xenon.workflow.ml_workflow import ML_Workflow

# lazy import dsmac
try:
    from dsmac.runhistory.runhistory_db import RunHistoryDB
    from dsmac.runhistory.utils import get_id_of_config
except Exception:
    RunHistoryDB = get_id_of_config = None


class TrainEvaluator(BaseEvaluator):
    def __init__(self):
        # ---member variable----
        self.debug = False

    def init_data(
            self,
            random_state,
            data_manager: DataManager,
            metric: Scorer,
            groups: List[int],
            should_calc_all_metric: bool,
            splitter,
            should_store_intermediate_result: bool,
            should_stack_X: bool,
            resource_manager: ResourceManager,
            should_finally_fit: bool,
            model_registry: dict,
            instance_id: str
    ):
        self.instance_id = instance_id
        self.should_stack_X = should_stack_X
        self.groups = groups
        if model_registry is None:
            model_registry = {}
        self.model_registry = model_registry
        self.random_state = random_state
        if not hasattr(splitter, "random_state"):
            setattr(splitter, "random_state", 42)  # random state is required, and is certain
        self.splitter = splitter
        self.data_manager = data_manager
        self.X_train = self.data_manager.X_train
        self.y_train = self.data_manager.y_train
        self.X_test = self.data_manager.X_test
        self.y_test = self.data_manager.y_test
        self.should_store_intermediate_result = should_store_intermediate_result
        self.metric = metric
        self.ml_task: MLTask = self.data_manager.ml_task

        self.should_calc_all_metric = should_calc_all_metric

        if self.ml_task.mainTask == "regression":
            self.predict_function = self._predict_regression
        else:
            self.predict_function = self._predict_proba

        self.logger = get_logger(self)
        self.resource_manager = resource_manager
        self.should_finally_fit = should_finally_fit

    def loss(self, y_true, y_hat):
        score, true_score = calculate_score(
            y_true, y_hat, self.ml_task.mainTask, self.metric,
            should_calc_all_metric=self.should_calc_all_metric)
        if isinstance(score, dict):
            err = self.metric._optimum - score[self.metric.name]
            all_score = true_score
        elif isinstance(score, (int, float)):
            err = self.metric._optimum - score
            all_score = None
        else:
            raise TypeError

        return err, all_score

    def set_resource_manager(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    def _predict_proba(self, X, model):
        y_pred = model.predict_proba(X)
        return y_pred

    def _predict_regression(self, X, model):
        y_pred = model.predict(X)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape((-1, 1))
        return y_pred

    def get_Xy(self):
        # fixme: ????????????????????????????????????
        #  ????????????bug???xenon.workflow.components.preprocessing.operate.keep_going.KeepGoing ?????????
        # fixme: xenon.manager.data_container.dataframe.DataFrameContainer#sub_sample ????????????deepcopy???
        #  ???????????????????????????X_train????????????????????????????????????X_test
        return (self.X_train), (self.y_train), (self.X_test), (self.y_test)
        # return deepcopy(self.X_train), deepcopy(self.y_train), deepcopy(self.X_test), deepcopy(self.y_test)

    def evaluate(self, model: ML_Workflow, X, y, X_test, y_test):
        assert self.resource_manager is not None
        warning_info = StringIO()
        additional_info = {}
        support_early_stopping = getattr(model.steps[-1][1], "support_early_stopping", False)
        with redirect_stderr(warning_info):
            # with open(__file__):
            # splitter ????????????
            losses = []
            models = []
            y_true_indexes = []
            y_preds = []
            y_test_preds = []
            all_scores = []
            status = "SUCCESS"
            failed_info = ""
            intermediate_results = []
            start_time = datetime.datetime.now()
            confusion_matrices = []
            best_iterations = []
            for train_index, valid_index in self.splitter.split(X.data, y.data, self.groups):
                cloned_model = model.copy()
                X: DataFrameContainer
                X_train = X.sub_sample(train_index)
                X_valid = X.sub_sample(valid_index)
                y_train = y.sub_sample(train_index)
                y_valid = y.sub_sample(valid_index)
                try:
                    procedure_result = cloned_model.procedure(self.ml_task, X_train, y_train, X_valid, y_valid, X_test,
                                                              y_test)
                except Exception as e:
                    failed_info = get_trance_back_msg()
                    status = "FAILED"
                    if self.debug:
                        self.logger.error("re-raise exception")
                        raise sys.exc_info()[1]
                    break
                intermediate_results.append(cloned_model.intermediate_result)
                models.append(cloned_model)
                y_true_indexes.append(valid_index)
                y_pred = procedure_result["pred_valid"]
                y_test_pred = procedure_result["pred_test"]
                if self.ml_task.mainTask == "classification":
                    confusion_matrices.append(calculate_confusion_matrix(y_valid.data, y_pred))
                if support_early_stopping:
                    estimator = cloned_model.steps[-1][1]
                    best_iterations.append(getattr(
                        estimator, "best_iteration_", getattr(estimator.component, "best_iteration_", -1)))
                # todo: ??????????????????y_pred
                y_preds.append(y_pred)
                if y_test_pred is not None:
                    y_test_preds.append(y_test_pred)
                loss, all_score = self.loss(y_valid.data, y_pred)  # todo: ???1d-array????????????????????????????????????
                losses.append(float(loss))
                all_scores.append(all_score)
            if self.ml_task.mainTask == "classification":
                additional_info["confusion_matrices"] = confusion_matrices
            if support_early_stopping:
                additional_info["best_iterations"] = best_iterations
            end_time = datetime.datetime.now()
            # finally fit
            if status == "SUCCESS" and self.should_finally_fit:
                # make sure have resource_manager to do things like connect redis
                model.resource_manager = self.resource_manager
                finally_fit_model = model.fit(X, y, X_test=X_test, y_test=y_test)
                if self.ml_task.mainTask == "classification":
                    y_test_pred_by_finally_fit_model = model.predict_proba(X_test)
                else:
                    y_test_pred_by_finally_fit_model = model.predict(X_test)
                model.resource_manager = None
            else:
                finally_fit_model = None
                y_test_pred_by_finally_fit_model = None

            if len(losses) > 0:
                final_loss = float(np.array(losses).mean())
            else:
                final_loss = 65535
            if len(all_scores) > 0 and all_scores[0]:
                all_score = defaultdict(list)
                for cur_all_score in all_scores:
                    if isinstance(cur_all_score, dict):
                        for key, value in cur_all_score.items():
                            all_score[key].append(value)
                    else:
                        self.logger.warning(f"TypeError: cur_all_score is not dict.\ncur_all_score = {cur_all_score}")
                for key in all_score.keys():
                    all_score[key] = float(np.mean(all_score[key]))
            else:
                all_score = {}
                all_scores = []
            info = {
                "loss": final_loss,
                "losses": losses,
                "all_score": all_score,
                "all_scores": all_scores,
                "models": models,
                "finally_fit_model": finally_fit_model,
                "y_true_indexes": y_true_indexes,
                "y_preds": y_preds,
                "intermediate_results": intermediate_results,
                "status": status,
                "failed_info": failed_info,
                "start_time": start_time,
                "end_time": end_time,
                "additional_info": additional_info
            }
            # todo
            if y_test is not None:
                # ?????????????????????????????????????????????????????????
                if self.should_finally_fit:
                    y_test_pred = y_test_pred_by_finally_fit_model
                else:
                    if self.ml_task.mainTask == "classification":
                        y_test_pred = vote_predicts(y_test_preds)
                    else:
                        y_test_pred = mean_predicts(y_test_preds)
                test_loss, test_all_score = self.loss(y_test.data, y_test_pred)
                # todo: ???1d-array????????????????????????????????????
                info.update({
                    "test_loss": test_loss,
                    "test_all_score": test_all_score,
                    # "y_test_true": y_test,
                    "y_test_pred": y_test_pred
                })
        info["warning_info"] = warning_info.getvalue()
        return info

    def __call__(self, shp: Configuration):
        # 1. ???php??????model
        config_id = get_id_of_config(shp)
        start = time()
        dhp, model = self.shp2model(shp)
        # 2. ????????????
        X_train, y_train, X_test, y_test = self.get_Xy()
        # 3. ????????????
        info = self.evaluate(model, X_train, y_train, X_test, y_test)
        # 4. ?????????
        cost_time = time() - start
        info["config_id"] = config_id
        info["instance_id"] = self.instance_id
        info["run_id"] = RunHistoryDB.get_run_id(self.instance_id, config_id)
        # info["program_hyper_param"] = shp
        info["dict_hyper_param"] = dhp
        estimator = list(dhp.get(PHASE2, {"unk": ""}).keys())[0]
        info["estimator"] = estimator
        info["cost_time"] = cost_time
        info["additional_info"].update({
            "config_origin": getattr(shp, "origin", "unk")
        })
        self.resource_manager.insert_trial_record(info)
        return {
            "loss": info["loss"],
            "status": info["status"],
        }

    def shp2model(self, shp):
        shp2dhp = SHP2DHP()
        dhp = shp2dhp(shp)
        # todo : ???????????????????????????????????????????????????3??????????????????????????????????????????????????????????????????????????????????????????
        preprocessor = self.create_preprocessor(dhp)
        estimator = self.create_estimator(dhp)
        pipeline = concat_pipeline(preprocessor, estimator)
        return dhp, pipeline

    def parse_key(self, key: str):
        cnt = ""
        ix = 0
        for i, c in enumerate(key):
            if c.isdigit():
                cnt += c
            else:
                ix = i
                break
        cnt = int(cnt)
        key = key[ix:]
        # pattern = re.compile(r"(\{.*\})")
        # match = pattern.search(key)
        # additional_info = {}
        # if match:
        #     braces_content = match.group(1)
        #     _to = braces_content[1:-1]
        #     param_kvs = _to.split(",")
        #     for param_kv in param_kvs:
        #         k, v = param_kv.split("=")
        #         additional_info[k] = v
        #     key = pattern.sub("", key)
        # todo: ????????????????????????????????????dataframe.py??????
        if "->" in key:
            _from, _to = key.split("->")
            in_feature_groups = _from.split(",")[0]
            out_feature_groups = _to.split(",")[0]
        else:
            in_feature_groups, out_feature_groups = None, None
        if not in_feature_groups:
            in_feature_groups = None
        if not out_feature_groups:
            out_feature_groups = None
        return in_feature_groups, out_feature_groups

    def create_preprocessor(self, dhp: Dict) -> Optional[ML_Workflow]:
        preprocessing_dict: dict = dhp[PHASE1]
        pipeline_list = []
        for key, value in preprocessing_dict.items():
            name = key  # like: "cat->num"
            in_feature_groups, out_feature_groups = self.parse_key(key)
            sub_dict = preprocessing_dict[name]
            if sub_dict is None:
                continue
            preprocessor = self.create_component(sub_dict, PHASE1, name, in_feature_groups, out_feature_groups,
                                                 )
            pipeline_list.extend(preprocessor)
        if pipeline_list:
            return ML_Workflow(pipeline_list, self.should_store_intermediate_result, self.resource_manager)
        else:
            return None

    def create_estimator(self, dhp: Dict) -> ML_Workflow:
        # ?????????????????????????????????
        return ML_Workflow(self.create_component(dhp[PHASE2], PHASE2, self.ml_task.role),
                           self.should_store_intermediate_result, self.resource_manager)

    def _create_component(self, key1, key2, params):
        cls = get_class_object_in_pipeline_components(key1, key2, self.model_registry)
        component = cls(**params)
        # component.set_addition_info(self.addition_info)
        return component

    def create_component(self, sub_dhp: Dict, phase: str, step_name, in_feature_groups="all", out_feature_groups="all",
                         outsideEdge_info=None):
        pipeline_list = []
        assert phase in (PHASE1, PHASE2)
        packages = list(sub_dhp.keys())[0]
        params = sub_dhp[packages]
        packages = packages.split(SERIES_CONNECT_SEPARATOR_TOKEN)
        grouped_params = group_dict_items_before_first_token(params, SERIES_CONNECT_LEADER_TOKEN)
        if len(packages) == 1:
            if bool(grouped_params):
                grouped_params[packages[0]] = grouped_params.pop("single")
            else:
                grouped_params[packages[0]] = {}
        for package in packages[:-1]:
            preprocessor = self._create_component(PHASE1, package, grouped_params[package])
            preprocessor.in_feature_groups = in_feature_groups
            preprocessor.out_feature_groups = in_feature_groups
            pipeline_list.append([
                package,
                preprocessor
            ])
        key1 = PHASE1 if phase == PHASE1 else self.ml_task.mainTask
        hyperparams = grouped_params[packages[-1]]
        if outsideEdge_info:
            hyperparams.update(outsideEdge_info)
        component = self._create_component(key1, packages[-1], hyperparams)
        component.in_feature_groups = in_feature_groups
        component.out_feature_groups = out_feature_groups
        if not self.should_stack_X:
            setattr(component, "need_y", True)
        pipeline_list.append([
            step_name,
            component
        ])
        return pipeline_list
