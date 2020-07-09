#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
from pathlib import Path

from scripts.utils import EnvUtils

relations = {
    "search": ["display"],
    "display": [],
    "ensemble": [],
    "predict": [],
}

pos2image = {
    "local": "harbor.atompai.com/nitrogen/xenon:v3.0",
    "xbcp": "",
}
nitrogen_json_dir = Path(__file__).parent.parent / "nitrogen_json"
for key, include_list in relations.items():
    include_list.append("common")
    include_list.append(key)
    env_utils = EnvUtils()
    for include in include_list:
        env_utils.from_json(f"env_configs/{include}.json")
    variables = env_utils.variables
    for ignore in ["REG_WORKFLOW", "CLF_WORKFLOW"]:
        variables[ignore] = "None"
    variables = {k: str(v) for k, v in variables.items()}
    data = {
        "name": f"xenon_{key}_script",
        "description": f"xenon {key} script",
        "git_url": "git@bitbucket.org:xtalpi/xenon.git",
        "git_branch": "v3.0",
        "git_commit": "",
        "datasets": "",
        "command": f"python scripts/{key}.json",
        "use_gpu": 0,
    }
    data["env"]=variables
    for pos,docker_image in pos2image.items():
        data["docker_image"]=docker_image
        (nitrogen_json_dir/pos/f"{key}.json").write_text(
            json.dumps(data,indent=4)
        )
