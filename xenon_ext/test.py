#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : qichun tang
# @Contact    : tqichun@gmail.com
import pandas as pd
from joblib import load

if __name__ == '__main__':
    df = pd.read_csv("mock_data.csv")
    model = load("model.bz2")
    prediction = model.predict(df)
    print(prediction)
