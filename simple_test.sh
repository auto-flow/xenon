export USER_ID=2
# 如果token过期，联系银州等同学; 注意对token中的\$进行转义
export USER_TOKEN="8v\$NdlCVujOey#&194fK%7OwYc8FNsMY"
export MODEL_TYPE=clf
mkdir -p ~/savedpath
rm -rf ~/savedpath/*
export SAVEDPATH=$HOME/savedpath
export DATAPATH=$HOME/python_packages/xenon/tests/datasets/iris.csv
export ENSEMBLE_SIZE=3
export RANDOM_RUNS=9
export BAYES_RUNS=0
export TRAIN_TARGET_COLUMN_NAME=target
export EXTERNAL_DELIVERY=True
# 目的是用动态的SPECIFIC_TASK_TOKEN关闭热启动（热启动需要相同的SPECIFIC_TASK_TOKEN）
export SPECIFIC_TASK_TOKEN=s`date +%Y_%m_%d_%H_%M_%S`
echo `SPECIFIC_TASK_TOKEN=$SPECIFIC_TASK_TOKEN`
cd ~/python_packages/xenon
python scripts/search.py