import inspect
import os
from typing import Union, Optional, Dict, List, Any

import numpy as np
import pandas as pd
from frozendict import frozendict
from joblib import dump
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from ultraopt import fmin
from ultraopt.multi_fidelity import HyperBandIterGenerator
from ultraopt.optimizer import ETPEOptimizer

from autoflow import constants
from autoflow.constants import ExperimentType
from autoflow.data_container import DataFrameContainer
from autoflow.data_manager import DataManager
from autoflow.hdl.hdl_constructor import HDL_Constructor
from autoflow.metrics import r2, mcc
from autoflow.resource_manager.base import ResourceManager
from autoflow.utils.klass import instancing
from autoflow.utils.logging_ import get_logger, setup_logger
from autoflow.evaluation.train_evaluator import  TrainEvaluator


class AutoFlowEstimator(BaseEstimator):
    checked_mainTask = None

    def __init__(
            self,
            hdl_constructor: Union[HDL_Constructor, List[HDL_Constructor], None, dict] = None,
            resource_manager: Union[ResourceManager, str] = None,
            model_registry=None,
            random_state=42,
            log_path: str = "autoflow.log",
            log_config: Optional[dict] = None,
            min_budget=1 / 16,
            use_BOHB=False,
            eta=4,
            imbalance_threshold=2,
            highR_nan_threshold=0.5,
            highC_cat_threshold=0.5,
            consider_ordinal_as_cat=False,
            should_store_intermediate_result=False,
            should_finally_fit=False,
            should_calc_all_metrics=True,
            should_stack_X=True,
            opt_early_stop_rounds=64,  # small: 32; middle: 64; big: 128
            total_time_limit=3600 * 5,  # 5 小时
            n_iterations=500,
            **kwargs
    ):
        '''
        Parameters
        ----------
        tuner: :class:`autoflow.tuner.Tuner` or None
            ``Tuner`` if class who agent an abstract search process.

        hdl_constructor: :class:`autoflow.hdl.hdl_constructor.HDL_Constructor` or None
            ``HDL`` is abbreviation of Hyper-parameter Descriptions Language.

            It describes an abstract hyperparametric space that independent with concrete implementation.

            ``HDL_Constructor`` is a class who is responsible for translating dict-type ``DAG-workflow`` into ``H.D.L`` .

        resource_manager: :class:`autoflow.manager.resource_manager.ResourceManager` or None
            ``ResourceManager`` is a class manager computer resources such like ``file_system`` and ``data_base``.

        random_state: int
            random state

        log_path: path
            which file to store log, if is None, ``autoflow.log`` will be used.

        log_config: dict
            logging configuration

        highR_nan_threshold: float
            high ratio NaN threshold, you can find example and practice in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`

        highC_cat_threshold: float
            high ratio categorical feature's cardinality threshold, you caGn find example and practice in :class:`autoflow.hdl.hdl_constructor.HDL_Constructor`

        kwargs
            if parameters like ``tuner`` or ``hdl_constructor`` and ``resource_manager`` are passing None,

            you can passing kwargs to make passed parameter work. See the following example.

        ExamplesG
        ---------
        In this example, you can see a trick to seed kwargs parameters with out initializing
        :class:`autoflow.hdl.hdl_constructor.HDL_Constructor` or other class.

        In following example, user pass ``DAG_workflow`` and ``hdl_bank`` by key-work arguments method.
        And we can see  hdl_constructor is instanced by kwargs implicitly.

        >>> from autoflow import AutoFlowClassifier
        >>> classifier = AutoFlowClassifier(DAG_workflow={"num->target":["lightgbm"]},
        ...   hdl_bank={"classification":{"lightgbm":{"boosting_type":  {"_type": "choice", "_value":["gbdt","dart","goss"]}}}})
        AutoFlowClassifier(hdl_constructor=HDL_Constructor(
            DAG_workflow={'num->target': ['lightgbm']}
            hdl_bank_path=None
            hdl_bank={'classification': {'lightgbm': {'boosting_type': {'_type': 'choice', '_value': ['gbdt', 'dart', 'goss']}}}}
            included_classifiers=('adaboost', 'catboost', 'decision_tree', 'extra_trees', 'gaussian_nb', 'k_nearest_neighbors', 'liblinear_svc', 'lib...
        '''
        self.n_iterations = n_iterations
        self.use_BOHB = use_BOHB
        self.total_time_limit = total_time_limit
        self.opt_early_stop_rounds = opt_early_stop_rounds
        self.imbalance_threshold = imbalance_threshold
        self.eta = eta
        self.min_budget = min_budget
        self.should_stack_X = should_stack_X
        self.consider_ordinal_as_cat = consider_ordinal_as_cat
        if model_registry is None:
            model_registry = {}
        assert isinstance(model_registry, dict)
        for key, value in model_registry.items():
            assert inspect.isclass(value)
        self.model_registry = model_registry
        self.should_finally_fit = should_finally_fit
        self.should_store_intermediate_result = should_store_intermediate_result
        self.should_calc_all_metrics = should_calc_all_metrics
        self.log_config = log_config
        self.highR_nan_threshold = highR_nan_threshold
        self.highC_cat_threshold = highC_cat_threshold
        # ---logger------------------------------------
        self.log_path = os.path.expandvars(os.path.expanduser(log_path))
        setup_logger(self.log_path, self.log_config)
        self.logger = get_logger(self)
        # ---random_state-----------------------------------
        self.random_state = random_state
        # ---tuner-----------------------------------
        # ---hdl_constructor--------------------------
        hdl_constructor = instancing(hdl_constructor, HDL_Constructor, kwargs)
        self.hdl_constructor = hdl_constructor
        # ---resource_manager-----------------------------------
        self.resource_manager: ResourceManager = instancing(resource_manager, ResourceManager, kwargs)
        # ---member_variable------------------------------------
        self.estimator = None
        self.ensemble_estimator = None

    def fit(
            self,
            X_train: Union[np.ndarray, pd.DataFrame, DataFrameContainer, str],
            y_train=None,
            X_test: Union[np.ndarray, pd.DataFrame, DataFrameContainer, str] = None,
            y_test=None,
            groups=None,
            upload_type="fs",
            sub_sample_indexes=None,
            sub_feature_indexes=None,
            column_descriptions: Optional[Dict] = frozendict(),
            metric=None,
            splitter=None,
            specific_task_token="",
            additional_info: dict = frozendict(),
            dataset_metadata: dict = frozenset(),
            task_metadata: dict = frozendict(),
            fit_ensemble_params: Union[str, Dict[str, Any], None, bool] = "auto",
            is_not_realy_run=False,
    ):
        '''

        Parameters
        ----------
        X_train: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        y_train: :class:`numpy.ndarray` or :class:`pandas.Series` or str
        X_test: :class:`numpy.ndarray` or :class:`pandas.DataFrame` or None
        y_test: :class:`numpy.ndarray` or :class:`pandas.Series` or str
        column_descriptions: dict
            Description about each columns' feature_group, you can find full definition in :class:`autoflow.manager.data_manager.DataManager` .
        dataset_metadata: dict
            Dataset's metadata
        metric: :class:`autoflow.metrics.Scorer` or None
            If ``metric`` is None:

            if it's classification task, :obj:`autoflow.metrics.accuracy` will be used by default.

            if it's regressor task, :obj:`autoflow.metrics.r2` will be used by default.
        should_calc_all_metrics: bool
            If ``True``, all the metrics supported in current task will be calculated, result will be store in databbase.
        splitter: object
            Default is ``KFold(5, True, 42)`` object. You can pass this param defined by yourself or other package,
            like :class:`sklearn.model_selection.StratifiedKFold`.
        specific_task_token: str
        should_store_intermediate_result: bool
        additional_info: dict
        fit_ensemble_params: str, dict, None, bool
            If this param is None, program will not do ensemble.

            If this param is "auto" or True, the top 10 models will be integrated by stacking ensemble.
        Returns
        -------
        self
        '''
        self.upload_type = upload_type
        self.sub_sample_indexes = sub_sample_indexes
        self.sub_feature_indexes = sub_feature_indexes
        dataset_metadata = dict(dataset_metadata)
        additional_info = dict(additional_info)
        task_metadata = dict(task_metadata)
        column_descriptions = dict(column_descriptions)
        # build data_manager
        self.data_manager: DataManager = DataManager(
            self.resource_manager,
            X_train, y_train, X_test, y_test, dataset_metadata, column_descriptions, self.highR_nan_threshold,
            self.highC_cat_threshold, self.consider_ordinal_as_cat, upload_type,
        )
        if is_not_realy_run:
            return self
        # parse ml_task
        self.ml_task = self.data_manager.ml_task
        if self.checked_mainTask is not None:
            if self.checked_mainTask != self.ml_task.mainTask:
                if self.checked_mainTask == "regression":
                    self.ml_task = constants.regression_task
                    self.data_manager.ml_task = self.ml_task
                else:
                    self.logger.error(
                        f"This task is supposed to be {self.checked_mainTask} task ,but the target data is {self.ml_task}.")
                    raise ValueError
        # parse splitter
        self.groups = groups
        if splitter is None:
            if self.ml_task.mainTask == "classification":
                splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                splitter = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        assert hasattr(splitter, "split"), "Parameter 'splitter' should be a train-valid splitter, " \
                                           "which contain 'split(X, y, groups)' method."
        self.splitter = splitter
        # parse metric
        if metric is None:
            if self.ml_task.mainTask == "regression":
                metric = r2
            elif self.ml_task.mainTask == "classification":
                metric = mcc  # qsar需求
            else:
                raise NotImplementedError()
        self.metric = metric
        # get task_id, and insert record into "tasks.tasks" database
        self.resource_manager.insert_task_record(
            data_manager=self.data_manager, metric=metric, splitter=splitter,
            specific_task_token=specific_task_token, dataset_metadata=dataset_metadata, task_metadata=task_metadata,
            sub_sample_indexes=sub_sample_indexes, sub_feature_indexes=sub_feature_indexes)
        self.resource_manager.close_task_table()
        # store other params
        setup_logger(self.log_path, self.log_config)
        hdl_constructor = self.hdl_constructor
        hdl_constructor.run(
            self.data_manager,
            self.model_registry, imbalance_threshold=self.imbalance_threshold)
        hdl = hdl_constructor.get_hdl()
        self.hdl = hdl
        if is_not_realy_run:
            return
        # get hdl_id, and insert record into "{task_id}.hdls" database
        self.resource_manager.insert_hdl_record(hdl, hdl_constructor.hdl_metadata)
        self.resource_manager.close_hdl_table()
        # now we get task_id and hdl_id, we can insert current runtime information into "experiments.experiments" database
        experiment_config = {
            "should_stack_X": self.should_stack_X,
            "should_finally_fit": self.should_finally_fit,
            "should_calc_all_metric": self.should_calc_all_metrics,
            "should_store_intermediate_result": self.should_store_intermediate_result,
            "fit_ensemble_params": str(fit_ensemble_params),
            "highR_nan_threshold": self.highR_nan_threshold,
            "highC_cat_threshold": self.highC_cat_threshold,
            "consider_ordinal_as_cat": self.consider_ordinal_as_cat,
            "random_state": self.random_state,
            "log_path": self.log_path,
            "log_config": self.log_config,
        }
        experiment_type = ExperimentType.AUTO
        self.resource_manager.insert_experiment_record(experiment_type, experiment_config, additional_info)
        self.resource_manager.close_experiment_table()
        self.task_id = self.resource_manager.task_id
        self.hdl_id = self.resource_manager.hdl_id
        self.experiment_id = self.resource_manager.experiment_id
        self.logger.info(f"task_id:\t{self.task_id}")
        self.logger.info(f"hdl_id:\t{self.hdl_id}")
        self.logger.info(f"experiment_id:\t{self.experiment_id}")
        self.evaluator = TrainEvaluator()
        self.evaluator.init_data(
            self.random_state,
            self.data_manager,
            self.metric,
            self.groups,
            self.should_calc_all_metrics,
            self.splitter,
            self.should_store_intermediate_result,
            self.should_stack_X,
            self.resource_manager,
            self.should_finally_fit,
            self.model_registry,
            ""
        )
        self.run_tuner(hdl)
        return self

    def run_tuner(self, cs):
        self.resource_manager.init_trial_table()
        # 是否启用BOHB
        if self.use_BOHB:
            multi_fidelity_iter_generator = HyperBandIterGenerator(
                min_budget=self.min_budget, max_budget=1, eta=self.eta)
        else:
            multi_fidelity_iter_generator = None
        # 贝叶斯代理模型为ETPE
        optimizer = ETPEOptimizer(min_points_in_model=10)
        result = fmin(  # 此时已经对 shps 设置过 n_jobs_in_algorithm
            self.evaluator, cs, optimizer=optimizer,
            n_jobs=1,
            n_iterations=self.n_iterations,
            random_state=np.random.randint(0, 10000),
            multi_fidelity_iter_generator=multi_fidelity_iter_generator,
            limit_resource=True,
            verbose=1,
            initial_points=None,
            warm_start_strategy="resume",
            previous_result=None,  # 用于热启动
            total_time_limit=self.total_time_limit,
            early_stopping_rounds=self.opt_early_stop_rounds,
        )
        # print(result)
        savedpath = os.getenv("SAVEDPATH")
        if savedpath and os.path.exists(savedpath):
            dump(result, f"{savedpath}/optimization_result.pkl")
