#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import click
import requests

token_dir = f"{os.getenv('HOME')}/xenon/auth"
token_file = f"{token_dir}/config.json"
# url = "http://192.168.1.182:9901"
# export XENON_URL=http://192.168.1.182:9901
url = os.getenv("XENON_URL", "https://xacs.nitrogen.fun:9090")
common_headers = {
    'Content-Type': 'application/json',
    'accept': 'application/json',
}


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

    response = requests.post(f"{url}/api/v1/login", headers=common_headers,
                             json={"user": email, "password": password})
    if response.status_code != 200:
        raise ConnectionError(f"response.status_code = {response.status_code}")
    json_data = response.json()
    data = json_data["data"]
    # print(json_data)
    # print(data)
    user_token = data["user_token"]
    user_id = data["user_id"]
    Path(token_dir).mkdir(parents=True, exist_ok=True)
    if Path(token_file).exists():
        config: dict = json.loads(Path(token_file).read_text())
    else:
        config = {}
    config.update({
        "user_token": user_token,
        "user_id": user_id,
        "email": email,
    })
    Path(token_file).write_text(json.dumps(config))
    print("Login Success.")


@auth.command()
def token():
    """
    Get USER_ID USER_TOKEN
    """
    _token()


def _token():
    try:
        config = json.loads(Path(token_file).read_text())
    except Exception:
        raise Exception("You should login first")
    user_id = config["user_id"]
    user_token = config["user_token"]
    check_login_status(url, user_id, user_token, common_headers)


def utc2local(utc_dtm):
    # UTC 时间转本地时间（ +8:00 ）
    local_tm = datetime.fromtimestamp(0)
    utc_tm = datetime.utcfromtimestamp(0)
    offset = local_tm - utc_tm
    return utc_dtm + offset


def check_login_status(url, user_id, user_token, common_headers):
    headers = deepcopy(common_headers)
    headers.update({
        "user_id": str(user_id),
        "user_token": user_token
    })
    response = requests.get(f"{url}/api/v1/user", headers=headers)
    json_response = response.json()
    if "data" in json_response and bool(json_response["data"]):
        data = json_response["data"]
        issued_on = data["issued_on"]
        issued_on = datetime.strptime(issued_on, '%Y-%m-%d %H:%M:%S')
        expires_on = data["expires_on"]
        expires_on = datetime.strptime(expires_on, '%Y-%m-%d %H:%M:%S')
        print("Your Login status is OK !")
        print("Login Time :\t", utc2local(issued_on))
        print("Expire Time :\t", utc2local(expires_on))
        print("-" * 50)
        print("USER_ID:")
        print(user_id)
        print("USER_TOKEN:")
        print(user_token)
    else:
        print("Your Login status is error !")
        print("status_code :\t", response.status_code)
        print("code :\t", json_response.get("code"))
        print("message :\t", json_response.get("message"))


if __name__ == '__main__':
    _token()
