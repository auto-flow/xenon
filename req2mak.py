#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
from pathlib import Path

Makefile = "Makefile"
lines = Path(Makefile).read_text().split("\n")
packages = Path("requirements.txt").read_text().split("\n")
packages = [package for package in packages if package and (not package.startswith("#"))]
deleted_lines = []
found = False
token = "install_pip_deps"
mark = f"{token}:"
for line in lines:
    if line == mark:
        found = True
    elif found and line.startswith("\t"):
        pass
    elif found and not line.startswith("\t"):
        found = False
        deleted_lines.append(line)
    else:
        deleted_lines.append(line)
final_lines = deleted_lines + [''] + [mark] + \
              [f"\tpip install {package}" for package in packages]
Path(Makefile).write_text("\n".join(final_lines))

