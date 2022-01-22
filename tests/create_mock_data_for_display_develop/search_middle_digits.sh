# 确保在xenon顶层目录
# sh tests/create_mock_data_for_display_develop/search_middle_digits.sh
export PYTHONPATH=$PYTHONPATH:$PWD
export DATAPATH=tests/datasets/middle_digits.csv
export SAVEDPATH=savedpath/multi-classification/search_middle_digits
rm -rf $SAVEDPATH
mkdir -p $SAVEDPATH
export USER_ID=2
export USER_TOKEN="8v\$NdlCVujOey#&194fK%7OwYc8FNsMY"
export TRAIN_TARGET_COLUMN_NAME=target
export MODEL_TYPE=clf
export RANDOM_RUNS=10
export BAYES_RUNS=0
export N_JOBS_IN_ALGORITHM=1
export ENSEMBLE_SIZE=1
export SEARCH_THREAD_NUM=1
python scripts/search.py
