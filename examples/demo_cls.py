import joblib
import pandas as pd
from sklearn.model_selection import KFold

from autoflow import AutoFlowClassifier

# load data from csv file
train_df = pd.read_csv("data/train_classification.csv")
test_df = pd.read_csv("data/test_classification.csv")
# initial_runs  -- initial runs are totally random search, to provide experience for ETPE algorithm.
# run_limit     -- is the maximum number of runs.
# n_jobs        -- defines how many search processes are started.
# included_classifiers -- restrict the search space . lightgbm is the only classifier that needs to be selected
# per_run_time_limit -- restrict the run time. if a trial during 60 seconds, it is expired, should be killed.
trained_pipeline = AutoFlowClassifier(
    initial_runs=5, run_limit=10, n_jobs=1,
    included_classifiers=["lightgbm"],
    per_run_time_limit=60,n_iterations=1)
# describing meaning of columns. `id`, `target` and `ignore` all has specific meaning
# `id` is a column name means unique descriptor of each rows,
# `target` column in the dataset is what your model will learn to predict
# `ignore` is some columns which contains irrelevant information
column_descriptions = {
    "id": "PassengerId",
    "target": "Survived",
    "ignore": "Name"
}
# if not os.path.exists("autoflow_classification.bz2"):
# pass `train_df`, `test_df` and `column_descriptions` to classifier,
# if param `fit_ensemble_params` set as "auto", Stack Ensemble will be used
# ``splitter`` is train-valid-dataset splitter, in here it is set as 3-Fold Cross Validation
trained_pipeline.fit(
    X_train=train_df, X_test=test_df, column_descriptions=column_descriptions,
    fit_ensemble_params=False,
    splitter=KFold(n_splits=3, shuffle=True, random_state=42),
)

graph = trained_pipeline.hdl_constructor.draw_workflow_space()
