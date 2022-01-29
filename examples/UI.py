#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
# set PYTHONPATH=D:\project\xenon;D:\project\ultraopt
# https://towardsdatascience.com/how-to-build-a-simple-machine-learning-web-app-in-python-68a45a0e0291
# https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602/8
import streamlit as st

from autoflow.hdl.hdl_constructor import HDL_Constructor

st.write("""
# Automatic Machine Learning App
This app predicts the **Titanic Prediction** type!
""")



st.sidebar.header('ML-Workflow Setting')
def input_HDL():
    included_classifiers = st.sidebar.multiselect('Classifiers',
                                          ["extra_trees", "lightgbm", "logistic_regression", "random_forest",
                                           "tabular_nn"], ['lightgbm'])
    included_imputers = st.sidebar.multiselect('NaN Imputers', ("impute.simple", "impute.gbt"), ['impute.simple'])
    included_highC_cat_encoders = st.sidebar.multiselect('High Cardinality Categories Encoders',
                                                 ["encode.entity", "encode.ordinal", "encode.cat_boost"],
                                                 ["encode.ordinal"])
    included_cat_encoders = st.sidebar.multiselect('Low Cardinality Categories Encoders',
                                           ["encode.one_hot", "encode.ordinal"],
                                           ["encode.one_hot"])
    scalers = st.sidebar.multiselect('Numerical Scaler',
                             ["scale.standard", "operate.keep_going"],
                             ["operate.keep_going"])
    feature_engineering = st.sidebar.multiselect('Feature Engineering',
                                         ["operate.keep_going", "select.boruta", "generate.autofeat"],
                                         ["operate.keep_going"])
    balance_strategies = st.sidebar.multiselect('Balance Strategies', ("weight", "None"), ['None'])
    hdl_constructor=HDL_Constructor(
        balance_strategies=balance_strategies, included_classifiers=included_classifiers,
        included_imputers=included_imputers, included_highC_cat_encoders=included_highC_cat_encoders,
        included_cat_encoders=included_cat_encoders, num2normed_workflow={"num->normed": scalers},
        normed2final_workflow={"normed->final": feature_engineering}
    )
    return hdl_constructor

st.subheader('User Input parameters')
hdl_constructor=input_HDL()
# st.write(hdl_constructor)

n_iterations = st.number_input('Number of iterations', 0, 10000, 10, 1, )

if st.button('Run Application'):
    pass
