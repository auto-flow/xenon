#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
data = {}
knn = {'n_neighbors': {'_type': 'int_loguniform', '_value': [1, 100], '_default': 3},
       'weights': {'_type': 'choice', '_value': ['uniform', 'distance'], '_default': 'uniform'},
       'p': {'_type': 'choice', '_value': [1, 2], '_default': 2}, 'n_jobs': 1}
svm = {'C': {'_type': 'loguniform', '_value': [0.01, 10000], '_default': 1.0},
       'kernel': {'_type': 'choice', '_value': ['rbf', 'poly', 'sigmoid'], '_default': 'rbf'}
       }
data['num->target(choice)'] = {
    'knn': knn,
    'svm': svm
}
data = {
    'num->target(choice)': {
        'knn': {
            'n_neighbors': {'_type': 'int_loguniform', '_value': [1, 100]},
            'weights': {'_type': 'choice', '_value': ['uniform', 'distance']},
            'p': {'_type': 'choice', '_value': [1, 2]}
        }, 'svm': {
            'C': {'_type': 'loguniform', '_value': [0.01, 10000]},
            'kernel': {'_type': 'choice', '_value': ['rbf', 'poly', 'sigmoid']}
        }
    }
}
from ultraopt.hdl.viz import plot_hdl, plot_layered_dict
module=1
print(module)
# g = plot_hdl(data)
# g.view()
data = {
    'num->target': {
        'knn': {
            'n_neighbors': 3,
            'weights': 'uniform',
            'p': 2
        }
    }
}
g = plot_layered_dict(data)
g.view()
