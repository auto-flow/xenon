#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import datetime
import simplejson as json
import logging
# from requests.packages.urllib3.util.retry import Retry
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy
import numpy as np
import pandas as pd
import requests
from frozendict import frozendict
from requests.adapters import HTTPAdapter
from requests.sessions import Session
from urllib3 import Retry
from urllib3.exceptions import InsecureRequestWarning

warnings.filterwarnings("ignore", category=InsecureRequestWarning)

logger = logging.getLogger(__name__)

session = Session()
retry = Retry(connect=10, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if (obj is None) or isinstance(obj, (str, int, float, bool, list, tuple)):
            return json.JSONEncoder.default(self, obj)
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return str(obj)
        elif isinstance(obj, bytes):
            # todo: base64
            return obj.decode(encoding="utf-8")
        elif isinstance(obj, frozendict):
            return dict(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.bool):
            return bool(obj)
        elif isinstance(obj, numpy.float):
            return float(obj)
        elif isinstance(obj, set):
            return list(obj)
        else:
            return str(obj)


# class HttpClient():
def judge_state_code(response: requests.Response):
    if response.status_code != 200:
        logger.warning("response.status_code != 200")
        if response.status_code == 401:
            logger.error("Authentication failed, maybe token is expired! AutoFlow-SDK will exit...")
            sys.exit(-1)
        return False
    return True


def get_data_of_response(response: requests.Response):
    try:
        json_response = response.json()
    except:
        return {}
    return json_response.get("data", {})


def send_requests(db_params: dict, target: str, json_data: Optional[dict] = None,
                  params: Optional[dict] = None,
                  method: str = "post") -> requests.Response:
    params = deepcopy(params)
    url = db_params["url"] + "/api/v1/" + target
    headers = db_params["headers"]
    kwargs = {
        "url": url,
        "headers": headers,
    }
    if params is not None:
        for k, v in params.items():
            if not isinstance(v, str):
                try:
                    sv = json.dumps(v)
                except Exception:
                    logger.debug(f"Cannot json.dump , key = {k}, value = {v}")
                    sv = str(v)
            else:
                sv = v
            params[k] = sv
        kwargs["params"] = params
    # fixme: 在这里处理nan值的问题
    if json_data is not None:
        json_data = json.dumps(json_data, cls=CustomJsonEncoder, ignore_nan=True, encoding="utf-8")
        kwargs.update({"data": json_data})
    kwargs.update(verify=False)
    # default values
    requests_exceptions = None
    response = None
    ok = False
    json_response = {}
    text = None
    # send requests
    try:
        if method == "post":
            response = session.post(**kwargs)
        elif method == "get":
            response = session.get(**kwargs)
        elif method == "delete":
            response = session.delete(**kwargs)
        elif method == "patch":
            response = session.patch(**kwargs)
        elif method == "put":
            response = session.put(**kwargs)
        else:
            raise NotImplementedError
    except Exception as e:
        logger.warning(f"Requests sending failed. exceptions:  {e}")
        requests_exceptions = str(e)
    if response is not None:
        ok = judge_state_code(response)
        try:
            json_response: Optional[dict] = response.json()
            text = None
        except Exception:
            json_response = {}
        text = response.text
    status_code = -1
    if response is not None and hasattr(response, "status_code"):
        status_code = response.status_code
    if requests_exceptions or json_response.get("code") != "1" or (not ok):
        if requests_exceptions:
            err_info = requests_exceptions
        elif response is not None and not ok:
            err_info = f"request url {url} status_code = {status_code} ."
        else:
            err_info = f"request url {url} response code != 1 ."
        log_file_name = f"{datetime.datetime.now()}.json"
        logger.warning(f"{err_info} | log_file_name = {log_file_name}")
        savedpath = os.getenv("SAVEDPATH")
        if savedpath is not None:
            root_path = savedpath
        else:
            root_path = f"{os.getenv('HOME')}/autoflow"
        log_path = f"{root_path}/requests_err"
        Path(log_path).mkdir(parents=True, exist_ok=True)
        log_file = f"{log_path}/{log_file_name}"
        err_data = {
            "status_code": status_code,
            "url": url,
            "db_params": db_params,
            "target": target,
            "json_data": json_data,
            "json_response": json_response,
            "text": text,
            "params": params,
            "method": method,
        }
        if requests_exceptions:
            err_data.update({"requests_exceptions": requests_exceptions})
        Path(log_file).write_text(json.dumps(err_data))
    return response


def extend_to_list(dict_: dict):
    if isinstance(dict_, dict):
        if not dict_:
            return []
        else:
            return [dict_]
    return dict_
