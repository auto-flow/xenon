# 确保在xenon顶层目录
# sh tests/regression_test/v1.2_predict.sh
export PYTHONPATH=$PYTHONPATH:$PWD
git checkout v1.2
export DATAPATH=tests/datasets/binary_iris.csv
export TASK_ID=`python -c 'import json5;data=json5.load(open("savedpath/regression_test/search/info.json"));print(data["task_id"])'`
export EXPERIMENT_ID=`python -c 'import json5;data=json5.load(open("savedpath/regression_test/search/info.json"));print(data["experiment_id"])'`
echo TASK_ID = ${TASK_ID}
echo EXPERIMENT_ID = ${EXPERIMENT_ID}
export SAVEDPATH=savedpath/regression_test/predict
mkdir -p $SAVEDPATH
export USER_ID=2
export USER_TOKEN="8v\$NdlCVujOey#&194fK%7OwYc8FNsMY"
python scripts/predict.py
