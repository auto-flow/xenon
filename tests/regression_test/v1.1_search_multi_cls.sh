# 确保在xenon顶层目录
# sh tests/regression_test/v1.1_search_multi_cls.sh
export PYTHONPATH=$PYTHONPATH:$PWD
git checkout v1.1
export DATAPATH=tests/datasets/small_digits.csv
export SAVEDPATH=savedpath/regression_test/search_search_multi_cls
rm -rf $SAVEDPATH
mkdir -p $SAVEDPATH
export USER_ID=2
export USER_TOKEN="8v\$NdlCVujOey#&194fK%7OwYc8FNsMY"
export TRAIN_TARGET_COLUMN_NAME=target
export MODEL_TYPE=clf
export RANDOM_RUNS=1
export BAYES_RUNS=1
export N_JOBS_IN_ALGORITHM=1
export ENSEMBLE_SIZE=1
export SEARCH_THREAD_NUM=1
python scripts/search.py
