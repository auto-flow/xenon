#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os
from pathlib import Path

import click
import requests

token_dir = f"{os.getenv('HOME')}/xenon/auth"
token_file = f"{token_dir}/config.json"
url = "http://192.168.1.182:9901"


@click.group()
def auth():
    pass


def load_email():
    try:
        return json.load(open(token_file))['email']
    except Exception:
        return None


@auth.command()
@click.option('--email', '-e', default='')
@click.option('--password', '-p', default='')
def login(email, password):
    """
    Login before use (valid in 24 hours)
    """
    if not email:
        default_email = load_email()
        msg = 'email'
        if default_email:
            msg = 'email({})'.format(default_email)
        email = click.prompt(msg, default='', show_default=False)
        if not email:
            email = default_email
    if not password:
        password = click.prompt('password', hide_input=True)
    headers = {
        'Content-Type': 'application/json',
        'accept': 'application/json',
    }
    response = requests.post(f"{url}/api/v1/login", headers=headers,
                             json={"user": email, "password": password})
    if response.status_code != 200:
        raise ConnectionError(f"response.status_code = {response.status_code}")
    json_data=response.json()
    data = json_data["data"]
    # print(json_data)
    # print(data)
    user_token = data["user_token"]
    user_id = data["user_id"]
    Path(token_dir).mkdir(parents=True, exist_ok=True)
    if Path(token_file).exists():
        config: dict = json.loads(Path(token_file).read_text())
    else:
        config={}
    config.update({
        "user_token": user_token,
        "user_id": user_id,
        "email": email,
    })
    Path(token_file).write_text(json.dumps(config))
    print("Login Success.")
