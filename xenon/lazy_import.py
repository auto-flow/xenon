#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import importlib

name2components = {
    "ConfigSpace": (
        "CategoricalHyperparameter", "Constant", "ConfigurationSpace",
        "ForbiddenInClause", "ForbiddenEqualsClause", "ForbiddenAndConjunction",
        "InCondition", "EqualsCondition", "UniformFloatHyperparameter",
        "UniformIntegerHyperparameter", "OrdinalHyperparameter", "Configuration"
    ),
    "hyperopt": ("fmin", "tpe", "hp", "space_eval")
}

for name, components in name2components.items():
    for component in components:
        try:
            globals()[component] = getattr(importlib.import_module(name), component)
        except Exception:
            globals()[component] = None

if __name__ == '__main__':
    print(UniformIntegerHyperparameter)
