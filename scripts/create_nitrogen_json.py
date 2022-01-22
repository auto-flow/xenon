#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from pathlib import Path

import pandas as pd

from scripts.utils import EnvUtils

relations = {
    "search": ["create_model"],
    "display": [],
    "ensemble": ["create_model"],
    "predict": [],
}

pos2image = {
    "local": "harbor.atompai.com/nitrogen/xenon:v3.0",
    "xbcp": "477093822308.dkr.ecr.us-east-2.amazonaws.com/nitrogen-1/xenon:v3.0",
}
docs_dir = Path(__file__).parent.parent / "docs"
env_table_dir = docs_dir / "env_table"
env_table_dir.mkdir(exist_ok=True, parents=True)
nitrogen_json_dir = docs_dir / "nitrogen_job_temp"
complex_env_dir = docs_dir / "complex_env"
nitrogen_json_dir.mkdir(exist_ok=True, parents=True)
complex_env_dir.mkdir(exist_ok=True, parents=True)
nitrogen_job_temp_rst = docs_dir / "nitrogen_job_temp.rst"
env_table_rst = docs_dir / "env_table.rst"
nitrogen_job_temp_rst.write_text("""

Nitrogen Job Templates
================================

""")
env_table_rst.write_text("""

ENV Description Table
================================

""")
ignore_list = ["REG_WORKFLOW", "CLF_WORKFLOW"]
for key, include_list in relations.items():
    include_list.insert(0, key)  # 第二
    include_list.insert(0, "common")  # 第一
    env_utils = EnvUtils()
    for include in include_list:
        env_utils.from_json(f"env_configs/{include}.json")
    variables = env_utils.variables
    env_items = env_utils.env_items
    for env_item in env_items:
        name = env_item["name"]
        if name in ignore_list:
            json_data = env_item["default"]
            name = name.lower()
            (complex_env_dir / f"{name}.json").write_text(json.dumps(json_data, indent=4))
            env_item["default"] = f":ref:`{name}`"
        env_item["default"] = str(env_item["default"])
    df = pd.DataFrame(env_items)
    df["default"] = df["default"].astype(str)
    df = df[["name", "default", "description"]]
    df.to_csv(env_table_dir / f"{key}.csv", index=False)
    for ignore in ignore_list:
        if ignore in variables:
            variables[ignore] = "None"
    variables = {k: str(v) for k, v in variables.items()}
    data = {
        "name": f"xenon_{key}_script",
        "description": f"xenon {key} script",
        "git_url": "git@bitbucket.org:xtalpi/xenon.git",
        "git_branch": "v3.0",
        "git_commit": "",
        "datasets": "",
        "command": f"python scripts/{key}.py",
        "use_gpu": 0,
    }
    data["env"] = variables
    for pos, docker_image in pos2image.items():
        data["docker_image"] = docker_image
        (nitrogen_json_dir / pos).mkdir(exist_ok=True, parents=True)
        (nitrogen_json_dir / pos / f"{key}.json").write_text(
            json.dumps(data, indent=4)
        )

f = nitrogen_job_temp_rst.open("a")
# [LOCAL XBCP]
for pos in pos2image:
    POS = pos.upper()
    f.write(f"""

{POS}
----------------------------------

""")
    # [search display ...
    for process in relations:
        Process = process.capitalize()
        f.write(f"""

{POS} {Process}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download template script for {Process}-Stage <nitrogen_job_temp/{pos}/{process}.json>`.

.. literalinclude:: nitrogen_job_temp/{pos}/{process}.json
   :language: json

如果你想了解 **{process}步骤** 的使用实例，可以跳转 :ref:`{Process} Stage`

""")
# 如果你想了解 **{process}步骤** 的使用实例，可以跳转 :ref:`{Process}`

f.close()

f = env_table_rst.open("a")

for process in relations:
    Process = process.capitalize()
    f.write(f"""

{Process} ENV Table
----------------------------------

.. csv-table:: {Process} ENV Table
   :file: env_table/{process}.csv

如果你想了解 **{process}步骤** 的使用实例，可以跳转 :ref:`{Process} Stage`

""")
    # 如果你想了解 **{process}步骤** 的使用实例，可以跳转 :ref:`{Process}`
# 开始构造复杂环境变量的描述
f.write("""

Complex ENV
----------------------------------

""")
for ignore in ignore_list:
    ignore = ignore.lower()
    f.write(f"""

{ignore}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`Download Complex ENV {ignore} <complex_env/{ignore}.json>`.

.. literalinclude:: complex_env/{ignore}.json
   :language: json

""")

f.close()
