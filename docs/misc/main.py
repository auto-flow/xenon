import os

import pandas as pd
from sklearn.datasets import load_digits

from scripts.search import search

# 指定中间文件路径
workdir = "/tmp"
datapath = f"{workdir}/data.csv"

# 数据生成与存储
X, y = load_digits(n_class=2, return_X_y=True)
df = pd.DataFrame(X)
df.columns = [f"col_{i}" for i in range(X.shape[1])]
df["target"] = y
df.to_csv(datapath, index=False)

# 指定 search 的环境变量
os.environ["TRAIN_TARGET_COLUMN_NAME"] = "target"
os.environ["MODEL_TYPE"] = "clf"

# 调用search
xenon_workflow = search(
    datapath=datapath,
    save_in_savedpath=True
)
