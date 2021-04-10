#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Date    : 2021-04-04
# @Contact    : qichun.tang@bupt.edu.cn
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np
from sklearn.utils.multiclass import type_of_target


def get_decision_tree_boundary(
        x: np.ndarray, y: np.ndarray,
        max_leaf_nodes=6,
        min_samples_leaf=0.05,
        min_max_bound=False,
        task=None
) -> np.ndarray:
    '''
        利用决策树获得最优分箱的边界值列表
    '''
    assert y is not None, ValueError
    boundary = []  # 待return的分箱边界值列表

    # x = x.fillna(nan).values  # 填充缺失值
    # y = y.values
    # 6 个叶子
    if task is None:
        task = "regression" if type_of_target(y) == "continuous" else "classification"
    if task == "classification":
        tree = DecisionTreeClassifier(
            criterion='entropy',  # “信息熵”最小化准则划分
            max_leaf_nodes=max_leaf_nodes,  # 最大叶子节点数
            min_samples_leaf=min_samples_leaf,  # 叶子节点样本数量最小占比
            random_state=0
        )
    else:
        tree = DecisionTreeRegressor(
            criterion="friedman_mse",
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            random_state=0
        )
    tree.fit(x.reshape(-1, 1), y)  # 训练决策树

    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    threshold = tree.tree_.threshold

    for i in range(n_nodes):
        if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
            boundary.append(threshold[i])

    boundary.sort()

    min_x = x.min()
    max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
    if min_max_bound:
        boundary = [min_x] + boundary + [max_x]
    return np.array(boundary)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    X, y = load_iris(True)
    boundary = get_decision_tree_boundary(X[:, 0], y)
    print(boundary)
