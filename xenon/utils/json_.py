#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import datetime
import json


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return str(obj)
        elif isinstance(obj, bytes):
            # todo: base64
            return obj.decode(encoding="utf-8")
        else:
            return json.JSONEncoder.default(self, obj)
