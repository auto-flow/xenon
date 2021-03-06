#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import datetime
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Tuple

import requests

from xenon import ResourceManager
from xenon.utils import list_
from xenon.utils.klass import get_valid_params_in_kwargs
from xenon.utils.logging_ import get_logger
from generic_fs.utils.http import send_requests, extend_to_list, get_data_of_response


def utc2local(utc_dtm):
    # UTC 时间转本地时间（ +8:00 ）
    local_tm = datetime.datetime.fromtimestamp(0)
    utc_tm = datetime.datetime.utcfromtimestamp(0)
    offset = local_tm - utc_tm
    return utc_dtm + offset


def check_login_status(url, user_id, user_token, common_headers, logger=None):
    headers = deepcopy(common_headers)
    if logger is None:
        func = print
    else:
        func = logger.info
    headers.update({
        "user_id": str(user_id),
        "user_token": user_token
    })
    response = requests.get(f"{url}/api/v1/user", headers=headers)
    json_response = response.json()
    if "data" in json_response and bool(json_response["data"]):
        data = json_response["data"]
        issued_on = data["issued_on"]
        issued_on = datetime.datetime.strptime(issued_on, '%Y-%m-%d %H:%M:%S')
        expires_on = data["expires_on"]
        expires_on = datetime.datetime.strptime(expires_on, '%Y-%m-%d %H:%M:%S')
        func("Your Login status is OK !")
        func(f"Login Time :\t{utc2local(issued_on)}")
        func(f"Expire Time :\t{utc2local(expires_on)}")
    else:
        func("Your Login status is Error !")
        func(f"status_code :\t{response.status_code}")
        func(f"code :\t{json_response.get('code')}")
        func(f"message :\t{json_response.get('message')}")
        if os.getenv("AUTH_ERR") == "ignore":
            pass
        else:
            sys.exit(-1)


class HttpResourceManager(ResourceManager):
    def __init__(
            self,
            url=None,
            email=None,
            password=None,
            user_id=None,
            user_token=None,
    ):

        if url is None:
            # url = "http://192.168.1.182:9901"
            # XENON_URL=https://xenon-test.nitrogen.fun:9091
            url = os.getenv("XENON_URL", "https://xenon.nitrogen.fun:9091")
        # todo: 增加encrypt字段
        self.url = url
        self.user_token = user_token
        self.user_id = user_id
        self.password = password
        self.email = email
        self.db_params = {
            "http_client": True,
            "url": url,
            "headers": {
                'Content-Type': 'application/json',
                'accept': 'application/json',
            }
        }
        token_dir = f"{os.getenv('HOME')}/xenon/auth"
        token_file = f"{token_dir}/config.json"
        self.login_logger = get_logger("Login")
        if email is None or password is None:
            self.login_logger.info("'email' or 'password' is None, try to "
                                   "verify User Authentication by 'user_id' and 'user_token'.")
            if user_id is None or user_token is None:
                self.login_logger.info("'user_id' and 'user_token' is None, "
                                       f"try to load token file '{token_file}'")
                if not Path(token_file).exists():
                    self.login_logger.error(f"user_token file '{token_file} do not exists! Xenon-SDK will exit...")
                    sys.exit(-1)
                config_data = json.loads(Path(token_file).read_text())
                if "user_token" not in config_data or "user_id" not in config_data:
                    self.login_logger.error(
                        f"'user_token' and 'user_id' did not exist in '{token_file}'! Xenon-SDK will exit...")
                    sys.exit(-1)
                self.user_token = config_data["user_token"]
                self.user_id = config_data["user_id"]
            self.db_params["headers"].update({"user_id": str(self.user_id), "user_token": self.user_token})
        else:
            self.db_params["user"] = self.email
            self.db_params["password"] = self.password
            self.user_id, self.user_token = self.login()
            Path(token_dir).mkdir(parents=True, exist_ok=True)
            Path(token_file).write_text(json.dumps({"user_id": self.user_id, "user_token": self.user_token}))
        check_login_status(self.url, self.user_id, self.user_token, self.db_params["headers"], self.login_logger)
        super(HttpResourceManager, self).__init__(
            store_path="xenon",
            db_params=self.db_params,
            user_id=self.user_id,
            file_system="nitrogen",
            file_system_params={
                "db_params": self.db_params
            },
            del_local_log_path=False
        )

    def login(self) -> Tuple[int, str]:
        user = self.db_params["user"]
        password = self.db_params["password"]
        response = send_requests(self.db_params, "login", {"user": user, "password": password})
        response_json = response.json()
        data = response_json["data"]
        self.user_token = data["user_token"]
        self.user_id = data["user_id"]
        self.db_params["headers"].update({
            "user_id": str(self.user_id),
            "user_token": self.user_token,
        })
        return self.user_id, self.user_token

    ##################################################################
    #########################  disable db connection #################
    ##################################################################
    def init_dataset_db(self):
        return None

    def init_record_db(self):
        return None

    def init_dataset_table(self):
        return None

    def init_hdl_table(self):
        return None

    def init_experiment_table(self):
        return None

    def init_task_table(self):
        return None

    def init_trial_table(self):
        return None

    ##################################################################
    ############################  dataset ############################
    ##################################################################

    def _insert_dataset_record(
            self,
            user_id: int,
            dataset_id: str,
            dataset_metadata: Dict[str, Any],
            dataset_type: str,
            dataset_path: str,
            upload_type: str,
            dataset_source: str,
            column_descriptions: Dict[str, Any],
            columns_mapper: Dict[str, str],
            columns: List[str]
    ):
        local = get_valid_params_in_kwargs(self._insert_dataset_record, locals())
        local.pop("user_id")
        local["columns"] = json.dumps(local["columns"])
        target = "dataset"
        response = send_requests(self.db_params, target, local)
        data = get_data_of_response(response)
        if "dataset_id" not in data:
            data["dataset_id"] = dataset_id
            data["dataset_path"] = dataset_path
            data["length"] = 0
        else:
            data["length"] = 1
        return data

    def _get_dataset_records(self, dataset_id, user_id) -> List[Dict[str, Any]]:
        target = f"dataset/{dataset_id}"
        response = send_requests(self.db_params, target, method="get")
        data = get_data_of_response(response)
        return extend_to_list(data)

    ##################################################################
    ###########################  experiment ##########################
    ##################################################################

    def _insert_experiment_record(
            self, user_id: int, hdl_id: str, task_id: str,
            experiment_type: str,
            experiment_config: Dict[str, Any], additional_info: Dict[str, Any]
    ):
        local = get_valid_params_in_kwargs(self._insert_experiment_record, locals())
        local.pop("user_id")
        target = "experiment"
        response = send_requests(self.db_params, target, local)
        data = get_data_of_response(response)
        if "experiment_id" not in data:
            raise ValueError("insert experiment failed.")
        return data["experiment_id"]

    def _finish_experiment_update_info(self, experiment_id: int, final_model_path: str, log_path: str,
                                       end_time: datetime.datetime):
        local = get_valid_params_in_kwargs(self._finish_experiment_update_info, locals())
        local.pop("experiment_id")
        target = f"experiment/{experiment_id}"
        response = send_requests(self.db_params, target, local, method="patch")

    def _get_experiment_record(self, experiment_id):
        target = f"experiment/{experiment_id}"
        response = send_requests(self.db_params, target, method="get")
        data = get_data_of_response(response)
        return extend_to_list(data)

    ##################################################################
    ############################   task    ###########################
    ##################################################################

    def _insert_task_record(self, task_id: str, user_id: int,
                            metric: str, splitter: Dict[str, str], ml_task: Dict[str, str],
                            train_set_id: str, test_set_id: str, train_label_id: str, test_label_id: str,
                            specific_task_token: str, task_metadata: Dict[str, Any], sub_sample_indexes: List[int],
                            sub_feature_indexes: List[str]):
        local = get_valid_params_in_kwargs(self._insert_task_record, locals())
        local["sub_sample_indexes"] = json.dumps(local["sub_sample_indexes"])
        local["sub_feature_indexes"] = json.dumps(local["sub_feature_indexes"])
        local.pop("user_id")
        target = "task"
        send_requests(self.db_params, target, local)
        return task_id

    def _get_task_records(self, task_id: str, user_id: int):
        target = f"task/{task_id}"
        response = send_requests(self.db_params, target, method="get")
        data = get_data_of_response(response)
        return extend_to_list(data)

    ##################################################################
    ############################   hdl     ###########################
    ##################################################################

    def _insert_hdl_record(self, task_id: str, hdl_id: str, user_id: int, hdl: dict, hdl_metadata: Dict[str, Any]):
        local = get_valid_params_in_kwargs(self._insert_hdl_record, locals())
        local.pop("user_id")
        local.pop("task_id")  # fixme: 没加 task_id
        target = "hdl"
        response = send_requests(self.db_params, target, local)
        return hdl_id

    ##################################################################
    ############################   trial   ###########################
    ##################################################################

    def _insert_trial_record(self, user_id: int, task_id: str, hdl_id: str, experiment_id: int, info: Dict[str, Any]):
        local = get_valid_params_in_kwargs(self._insert_trial_record, locals())
        local.pop("user_id")
        target = "trial"
        info = local.pop("info")
        local.update(**info)
        local.pop("hdl_id")
        local.pop("task_id")
        local.pop("finally_fit_model", None)
        local["final_model_path"] = local.pop("finally_fit_model_path", None)
        # local["intermediate_results"]=json.dumps(local["intermediate_results"])
        local["losses"] = json.dumps(local["losses"])
        local["all_scores"] = json.dumps(local["all_scores"])
        # fixme
        # local["test_loss"]={}
        # local["test_all_score"]={}
        response = send_requests(self.db_params, target, local)
        data = get_data_of_response(response)
        if "trial_id" not in data:
            self.logger.warning("insert trial failed.")
        return data.get("trial_id", -1)

    def _get_sorted_trial_records(self, task_id, user_id, limit):
        target = f"task/{task_id}/trial?page_num={1}&page_size={limit}"
        response = send_requests(self.db_params, target, method="get")
        data = get_data_of_response(response)
        return extend_to_list(data)

    def _get_trial_records_by_id(self, trial_id, task_id, user_id):
        target = f"trial/{trial_id}"
        response = send_requests(self.db_params, target, method="get")
        # 返回值是一个字典
        data = get_data_of_response(response)
        return data

    def _get_trial_records_by_ids(self, trial_ids, task_id, user_id):
        result = []
        for sub_trial_ids in list_.chunk(trial_ids, 10):
            target = f"trial?" + "&".join([f"trial_id={trial_id}" for trial_id in sub_trial_ids])
            response = send_requests(self.db_params, target, method="get")
            data = get_data_of_response(response)
            if data:
                result.extend(data)
            else:
                break
        return result

    def _get_best_k_trial_ids(self, task_id, user_id, k):
        data = self._get_sorted_trial_records(task_id, user_id, k)
        return [item["trial_id"] for item in data]
