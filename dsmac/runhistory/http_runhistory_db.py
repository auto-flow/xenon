import datetime
import inspect
import json
from typing import Tuple, Optional, Dict, Any

import peewee as pw
import requests

from dsmac.runhistory.runhistory_db import RunHistoryDB
from generic_fs.utils.http import CustomJsonEncoder, send_requests, extend_to_list


def get_valid_params_in_kwargs(func, kwargs: Dict[str, Any]):
    validated = {}
    for key, value in kwargs.items():
        if key in inspect.signature(func).parameters.keys():
            validated[key] = value
    return validated


class HttpRunHistoryDB(RunHistoryDB):

    def get_db_cls(self):
        return None

    def get_db(self):
        return None

    def get_model(self) -> Optional[pw.Model]:
        return None


    ##################################################################
    #########################   run_history   ########################
    ##################################################################

    def _appointment_config(self, run_id, instance_id) -> Tuple[bool, Optional[pw.Model]]:
        local = get_valid_params_in_kwargs(self._appointment_config, locals())
        target = "history"
        response = send_requests(self.db_params,target, local)
        json_response = response.json()["data"]
        if "run_id" in json_response:  # 创建成功
            return True, None
        else:
            return False, None
        # return json_response["ok"], json_response["record"]

    def _insert_runhistory_record(
            self, run_id, config_id, config, config_origin, cost: float, time: float,
            status: int, instance_id: str,
            seed: int,
            additional_info: dict,
            origin: int,
            modify_time: str,
            pid: int,
    ):
        additional_info = dict(additional_info)
        modify_time = datetime.datetime.now()
        local = get_valid_params_in_kwargs(self._insert_runhistory_record, locals())
        target = f"history/{run_id}"
        response = send_requests(self.db_params,target, local, method="patch")
        return

    def _fetch_new_runhistory(self, instance_id, pid, timestamp, is_init):
        local = get_valid_params_in_kwargs(self._fetch_new_runhistory, locals())
        result = []
        page_num = 1
        page_size = 20
        while True:
            target = f"history?page_num={page_num}&page_size={page_size}"
            response = send_requests(self.db_params,target, params=local, method="get")
            data = extend_to_list(response.json()["data"])
            result.extend(data)
            if len(data) < page_size:
                break
            page_num += 1
        return result
