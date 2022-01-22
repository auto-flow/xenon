
# Xenon 安装说明

以下所有步骤请获取最新的 `v1.2` 版本代码

## xenon, xenon-ext, xenon-tiny 三者的不同点（功能上，适用场景和范围）详细说明

|| 文件名 | package名 | 功能 | 适用场景和范围 |
|-|----|---------|----|------------|
| xenon | requirements.txt | xenon | search+predict | 实现全部xenon功能，如线上renova xenon 环境 |
| xenon-tiny| requirements_tiny.txt | xenon | only predict | 在安装极少量依赖的情况下，实现最小可用的预测功能 |
| xenon-ext| requirements_ext.txt | xenon_ext | 在不依赖xenon代码的情况下做预测 | 对外交付场景 |



## 如何将 requirments.txt中的内容与Makefile中install_pip_deps保持一致

因为国内网络环境的原因，如果在不翻墙情况下希望下载Python package更快，需要换源，可在终端执行：

```bash
make change_pip_source 
```

但存在一个问题，如果需要安装xenon（假设是完整版，不是tiny），若使用 `pip install .` 或者 `python setup.py install` 命令，会默认不走代理，导致下载速度慢。

所以当时（2020年4-6月）我在解决这个问题时，采用先执行 `make install_pip_deps`， 再执行 `python setup.py install` 的方案，即先走代理将需要的package都装了，然后装xenon，解决了安装速度慢的问题。

所以，如果要将 requirments.txt中的内容与Makefile中install_pip_deps保持一致，可执行：

```bash
python req2mak.py
```

## 针对xenon, xenon-ext, xenon-tiny 的不同版本（v1.1\v1.2），打包不同的whl文件并通过测试

执行命令：

make bdist_wheel

whl 安装包生成于 whl 文件夹

测试：

```bash
cd whl_dist
virtualenv venv
source venv/bin/activate
pip install *.whl
```

测试通过


# xenon4bigdata 数据链路

pycharm 代码自动换行：

setting 搜 Soft Wraps，加一个快捷键

```bash
mkdir -p data
nitrogen download 103252
mv job_33483_result data/job_33483_result
mkdir -p savedpath
```

# Xenon3 search fit
mkdir -p savedpath/xenon3_search_fit
DATAPATH=data/job_33483_result;SAVEDPATH=savedpath/xenon3_search_fit;MODEL_TYPE=clf;TRAIN_TARGET_COLUMN_NAME=active;USER_ID=2;USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;METRIC=auc;SPECIFIC_TASK_TOKEN=test_xenon3;RANDOM_RUNS=1;BAYES_RUNS=1;



## 1. preprocess

DATAPATH=data/job_33483_result;SAVEDPATH=savedpath/xenon4bigdata/bd1_preprocess_clf;MODEL_TYPE=clf;TRAIN_TARGET_COLUMN_NAME=active;FEATURE_SELECT_METHOD=l1_linear;SELECTOR_PARAMS=l1_linear(C=1,alpha=0.001,max_iter=100),rf(n_estimators=1000),gbdt(n_estimators=100)

## 2. search 

XENON_URL=https://xenon-test.nitrogen.fun:9091;USER_ID=2;USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;DATAPATH=savedpath/xenon4bigdata/bd1_preprocess_clf;SAVEDPATH=savedpath/xenon4bigdata/bd2_search_gbdt_lr;MODEL_TYPE=clf;METRIC=auc;ESTIMATOR_PARAMS=lgbm_gbdt_lr(n_estimators=400, early_stopping_rounds=50);SPECIFIC_TASK_TOKEN=test_sys;N_ITERATIONS=10;ESTIMATOR_CHOICES=None;N_WORKERS=1

## 3. display

>task_id trial_id 都要看情况来设置

XENON_URL=https://xenon-test.nitrogen.fun:9091;USER_ID=2;USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;TASK_ID=5fcd5a953ee44b55cdf573b8ae49530d;DISPLAY_SIZE=20;SAVEDPATH=savedpath/xenon4bigdata/bd3_display

## 4. ensemble

>task_id trial_id 都要看情况来设置

XENON_URL=https://xenon-test.nitrogen.fun:9091;USER_ID=2;USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;TASK_ID=5fcd5a953ee44b55cdf573b8ae49530d;TRIAL_ID=[53788,53793,53790];SAVEDPATH=savedpath/xenon4bigdata/bd4_ensemble

## 5. predict

>experiment_id 根据ensemble结果来设置

XENON_URL=https://xenon-test.nitrogen.fun:9091;USER_ID=2;USER_TOKEN=8v$NdlCVujOey#&194fK%7OwYc8FNsMY;EXPERIMENT_ID=1470;DATAPATH=data/job_33483_result;SAVEDPATH=savedpath/xenon4bigdata/bd5_predict;TRAIN_TARGET_COLUMN_NAME=active