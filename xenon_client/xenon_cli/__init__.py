#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from .commond.auth import auth
import click

cli = click.CommandCollection(sources=[auth])