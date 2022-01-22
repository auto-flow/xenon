# 如何使用本文件夹下的模型

- 安装xenon_ext依赖


```bash
virtualenv venv
source venv/bin/activate
# 加 -i参数表示使用清华源进行安装，加快下载速度
pip install pip install xenon_ext-*.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
```

默认只安装最小依赖环境（`sklearn` + `lightgbm`）。对于有特殊依赖的模型，请：

```bash
pip install xgboost
pip install catboost
pip install xlearn
pip install torch 
pip install tensorflow  
... 
```

- 测试load `experiment_*_best_model.bz2` 模型

```bash
python external_delivery_test.py

```
