#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import ast
import json
import os
from pathlib import Path
from pprint import pprint

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
        print("--------------------")
        print("| Complex variable |")
        print("--------------------")

        for k, v in self.long_data:
            print(k + " : " + type(v).__name__)
            pprint(v)
            print("-" * 50)

    __repr__ = __str__
