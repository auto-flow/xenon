# -*- encoding: utf-8 -*-
import json
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Union, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from frozendict import frozendict
from sklearn.preprocessing import LabelEncoder

from xenon.constants import AUXILIARY_FEATURE_GROUPS, NAN_FEATURE_GROUPS, UNIQUE_FEATURE_GROUPS
from xenon.data_container import DataFrameContainer
from xenon.data_container import NdArrayContainer
from xenon.data_container.base import get_container_data
from xenon.utils.data import is_nan, is_cat, is_highR_nan, to_array, is_highR_cat, is_date, is_text, \
    is_target_need_label_encode
from xenon.utils.dataframe import get_unique_col_name
from xenon.utils.klass import StrSignatureMixin
from xenon.utils.logging_ import get_logger
from xenon.utils.ml_task import MLTask, get_ml_task_from_y
from xenon.workflow.components.utils import stack_Xs


def pop_if_exists(df: Union[pd.DataFrame, DataFrameContainer], col: str) -> Optional[pd.DataFrame]:
    if df is None:
        return None
    if isinstance(df, DataFrameContainer):
        df = df.data
    if col in df.columns:
        return df.pop(col)
    else:
        return None


class DataManager(StrSignatureMixin):
    '''
    DataManager is a Dataset manager to store the pattern of dataset.
    '''

    def __init__(
            self,
            resource_manager=None,
            X_train: Union[pd.DataFrame, DataFrameContainer, np.ndarray, None, str] = None,
            y_train: Union[pd.Series, np.ndarray, None] = None,
            X_test: Union[pd.DataFrame, DataFrameContainer, np.ndarray, None, str] = None,
            y_test: Union[pd.Series, np.ndarray, None] = None,
            dataset_metadata: Dict[str, Any] = frozendict(),
            column_descriptions: Dict[str, Union[List[str], str]] = frozendict(),
            highR_nan_threshold: float = 0.5,
            highR_cat_threshold: float = 0.5,
            consider_ordinal_as_cat=False,
            upload_type="fs",
            is_not_realy_run=False
    ):
        '''

        Parameters
        ----------
        X_train: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        y_train: :class:`numpy.ndarray`
        X_test: :class:`numpy.ndarray` or :class:`pandas.DataFrame`
        y_test: :class:`numpy.ndarray`
        dataset_metadata: dict
        column_descriptions: dict
            ``column_descriptions`` is a dict, key is ``feature_group``,

            value is column (column name) or columns (list of column names).

            This is a list of some frequently-used built-in ``feature_group``
                * ``id``       - id of this table.
                * ``ignore``   - some columns which contains irrelevant information.
                * ``target``   - column in the dataset is what your model will learn to predict.
                * ``nan``      - Not a Number, a column contain missing values.
                * ``num``      - numerical features, such as [1, 2, 3].
                * ``cat``      - categorical features, such as ["a", "b", "c"].
                * ``num_nan``  - numerical features contains missing values. such as [1, 2, NaN].
                * ``cat_nan``  - categorical features contains missing values. such as ["a", "b", NaN].
                * ``highR_nan``  - highly ratio NaN. You can find explain in :class:`xenon.hdl.hdl_constructor.HDL_Constructor`
                * ``lowR_nan``   - lowly ratio NaN. You can find explain in :class:`xenon.hdl.hdl_constructor.HDL_Constructor`
                * ``highR_cat``  - highly cardinality ratio categorical. You can find explain in :class:`xenon.hdl.hdl_constructor.HDL_Constructor`
                * ``lowR_cat``  -  lowly cardinality ratio categorical. You can find explain in :class:`xenon.hdl.hdl_constructor.HDL_Constructor`

        highR_nan_threshold: float
            high ratio NaN threshold, you can find examples and practice in :class:`xenon.hdl.hdl_constructor.HDL_Constructor`
        '''
        self.upload_type = upload_type
        from xenon.resource_manager.base import ResourceManager
        self.logger = get_logger(self)
        if resource_manager is None:
            self.logger.warning(
                "In DataManager __init__, resource_manager is None, create a default local resource_manager.")
            resource_manager = ResourceManager()
        self.resource_manager: ResourceManager = resource_manager
        self.resource_manager = resource_manager
        self.highR_cat_threshold = highR_cat_threshold
        self.consider_ordinal_as_cat = consider_ordinal_as_cat
        dataset_metadata = dict(dataset_metadata)
        self.highR_nan_threshold = highR_nan_threshold
        self.dataset_metadata = dataset_metadata
        self.column_descriptions = dict(column_descriptions)
        # --load data to container---------------------------------
        self.X_test, self.input_test_hash = self.parse_data_container("TestSet", X_test, y_test)
        #             train set ????????????train set ???column_descriptions??????
        self.X_train, self.input_train_hash = self.parse_data_container("TrainSet", X_train, y_train)
        # --migrate column descriptions------------------------------
        # if X is dataset_id , remote data_container's column_descriptions will assigned to  final_column_descriptions
        if self.final_column_descriptions is not None:
            self.column_descriptions = deepcopy(self.final_column_descriptions)
        # --column descriptions------------------------------
        self.parse_column_descriptions()
        # ???????????????feature_groups???columns????????????????????????????????????????????????
        # ---check target-----------------------------------------------------
        assert "target" in self.column_descriptions
        self.target_col_name = self.column_descriptions["target"]
        # todo: ????????????????????????
        # --final column descriptions------------------------------
        # ??????????????? column descriptions ??? remote ?????????column description??????????????????nan?????????
        # update `column2essential_feature_groups` to `final_column_descriptions`
        if self.final_column_descriptions is None:
            final_column_descriptions = defaultdict(list)
            final_column_descriptions.update(self.column_descriptions)
            # ??????????????????????????????????????????
            for feat_grp, cols in final_column_descriptions.items():
                if feat_grp not in UNIQUE_FEATURE_GROUPS:
                    if isinstance(cols, str):
                        final_column_descriptions[feat_grp] = [cols]
            # ??????????????????
            for column, essential_feature_group in self.column2essential_feature_groups.items():
                if column not in final_column_descriptions[essential_feature_group]:
                    final_column_descriptions[essential_feature_group].append(column)
            self.final_column_descriptions = final_column_descriptions
        self.final_column_descriptions = dict(self.final_column_descriptions)
        if not is_not_realy_run:
            # ---set column descriptions, upload to dataset-----------------------------------------------------
            if self.X_train is not None:
                self.X_train.set_column_descriptions(self.final_column_descriptions)
                self.X_train.upload(self.upload_type, upload_data=False)
                self.logger.info(f"TrainSet's DataSet ID = {self.X_train.dataset_id}")
            if self.X_test is not None:
                self.X_test.set_column_descriptions(self.final_column_descriptions)
                self.X_test.upload(self.upload_type, upload_data=False)
                self.logger.info(f"TestSet's DataSet ID = {self.X_test.dataset_id}")
            # ---origin hash-----------------------------------------------------
            self.train_set_id = self.X_train.get_hash() if self.X_train is not None else ""
            self.test_set_id = self.X_test.get_hash() if self.X_test is not None else ""
            if self.input_train_hash:
                assert self.input_train_hash == self.train_set_id
            if self.input_test_hash:
                assert self.input_test_hash == self.test_set_id
        # ---pop auxiliary columns-----------------------------------------------------
        y_train, y_test = self.pop_auxiliary_feature_groups()
        # --??????X???X_test??????????????????
        if self.X_test is not None and self.X_train is not None:
            assert self.X_train.shape[1] == self.X_test.shape[1]
            assert np.all(self.X_train.columns == self.X_test.columns)
        # --??????feature_groups--
        if self.X_train is not None:
            self.X_train.set_feature_groups(self.feature_groups)
        if self.X_test is not None:
            self.X_test.set_feature_groups(self.feature_groups)
        # --????????????--
        y_train = to_array(y_train)
        y_test = to_array(y_test)
        # encode label
        assert y_train is not None, f"{self.target_col_name} does not exist!"
        self.label_encoder = None
        if is_target_need_label_encode(y_train):
            self.label_encoder = LabelEncoder()
            y_train = self.label_encoder.fit_transform(y_train)
            y_test = self.encode_label(y_test)
        self.y_train = self.y_test = None
        if y_train is not None and not is_not_realy_run:
            y_train = NdArrayContainer("TrainLabel", dataset_instance=y_train,
                                       resource_manager=self.resource_manager)
            y_train.upload()
            self.y_train = y_train
            self.train_label_id = self.y_train.get_hash()
        else:
            self.y_train = None
            self.train_label_id = ""
        if y_test is not None and not is_not_realy_run:
            y_test = NdArrayContainer("TestLabel", dataset_instance=y_test,
                                      resource_manager=self.resource_manager)
            y_test.upload()
            self.y_test = y_test
            self.test_label_id = self.y_test.get_hash()
        else:
            self.y_test = None
            self.test_label_id = ""
        self.ml_task: MLTask = get_ml_task_from_y(y_train.data)
        if self.X_train is not None:
            self.columns = self.X_train.columns
        else:
            self.columns = self.X_test.columns

        # todo: ????????????????????????????????????RandomShuffle ??????mlxtend??????
        # fixme: ?????????multilabel

    def encode_label(self, y):
        if self.label_encoder is not None:
            try:
                return self.label_encoder.transform(y) if y is not None else None
            except Exception as e:
                return y
        return y

    def pop_auxiliary_feature_groups(self):
        y_train = pop_if_exists(self.X_train, self.target_col_name)
        y_test = pop_if_exists(self.X_test, self.target_col_name)
        # --??????id--
        if "id" in self.column_descriptions:
            id_col = self.column_descriptions["id"]  # id ??????????????????
            self.train_id_seq = pop_if_exists(self.X_train, id_col)
            self.test_id_seq = pop_if_exists(self.X_test, id_col)
        # --??????ignore--
        if "ignore" in self.column_descriptions:
            ignore_cols = self.column_descriptions["ignore"]
            if not isinstance(ignore_cols, (list, tuple)):
                ignore_cols = [ignore_cols]
            for ignore_col in ignore_cols:
                pop_if_exists(self.X_train, ignore_col)
                pop_if_exists(self.X_test, ignore_col)
        return y_train, y_test

    def concat_y(self, X, y):
        # if isinstance(y,)
        if isinstance(y, (np.ndarray, pd.Series)):
            # ??????y???target?????????X???????????????column_descriptions
            y = pd.Series(y)
            target_col_name = get_unique_col_name(X.columns, "target")
            y = pd.DataFrame(y, columns=[target_col_name])
            self.column_descriptions.update({"target": target_col_name})
            y.index = X.index
            assert y.shape[0] == X.shape[0]
            X = pd.concat([X, y], axis=1)
        return X

    def parse_data_container(self, dataset_source, X, y) -> Tuple[Optional[DataFrameContainer], str]:
        if X is None:
            return X, ""
        # input_dataset_id only work if X is dataset_id
        # keep input_dataset_id to do sample test
        # make sure dataset is invariant in upload and download process
        input_dataset_id = ""
        self.final_column_descriptions = None
        # filepath or dataset_id
        if isinstance(X, str):
            # filepath
            if os.path.exists(X):
                self.logger.info(f"'{X}' will be treated as a file path.")
                X = DataFrameContainer(dataset_source, dataset_path=X, resource_manager=self.resource_manager,
                                       dataset_metadata=self.dataset_metadata)
            # dataset_id
            else:
                self.logger.info(f"'{X}' will be treated as dataset ID, and download from database.")
                input_dataset_id = X
                X = DataFrameContainer(dataset_source, dataset_id=X, resource_manager=self.resource_manager,
                                       dataset_metadata=self.dataset_metadata)
                self.final_column_descriptions = deepcopy(X.column_descriptions)
        elif isinstance(X, DataFrameContainer):
            pass
        else:
            # we should create a columns and concat X and y
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X,
                                 columns=[f"column_{i}" for i in range((X.shape[1]))])
            # in this step, column_descriptions will implicitly update "target" field
            X = self.concat_y(X, y)
            X = DataFrameContainer(dataset_source, dataset_instance=X, resource_manager=self.resource_manager,
                                   dataset_metadata=self.dataset_metadata)
        return X, input_dataset_id

    def parse_feature_group(self, series: pd.Series, consider_nan=True) -> str:
        # --- start parsing feature-group -----
        if consider_nan and is_nan(series):
            if is_highR_nan(series, self.highR_nan_threshold):
                feature_group = "highR_nan"
            else:
                feature_group = "nan"
        elif is_cat(series, self.consider_ordinal_as_cat):
            if is_date(series, True):
                feature_group = "date"
            elif is_text(series, True):
                feature_group = "text"
            else:
                if is_highR_cat(series, self.highR_cat_threshold):
                    feature_group = "highR_cat"
                else:
                    feature_group = "cat"
        else:
            feature_group = "num"

        return feature_group

    def parse_column_descriptions(self):
        if self.X_train is None and self.X_test is None:
            self.logger.error("X_train and X_test are both None, it is invalide.")
            raise ValueError
        if self.column_descriptions is None:
            self.column_descriptions = {}
        # --??????????????????--
        userDefined_column2feature_groups = {}
        column2feature_groups = {}
        # if user defined column_descriptions or download dataset from remote
        # this process will gather `userDefined_column2feature_groups`
        for feat_grp, columns in self.column_descriptions.items():
            if isinstance(columns, str):
                columns = [columns]
            for column in columns:
                userDefined_column2feature_groups[column] = feat_grp
        # `column2feature_groups` and `column2essential_feature_groups` 's k-v will
        column2feature_groups.update(deepcopy(userDefined_column2feature_groups))
        column2essential_feature_groups = deepcopy(column2feature_groups)
        # ----?????????X_train???X_test??????????????????????????????---------
        X = stack_Xs(
            get_container_data(self.X_train),
            None,
            get_container_data(self.X_test)
        )  # fixme:target????????????nan
        # --??????????????????????????????nan--
        for column, feature_group in list(column2feature_groups.items()):
            # ???????????????feature_groups???columns????????????????????????????????????????????????
            if feature_group in AUXILIARY_FEATURE_GROUPS:  # ("id", "target", "ignore")
                continue
            nan_col = self.detect_nan_feature_group(X[column])
            if nan_col is not None:
                column2feature_groups[column] = nan_col
                # ???????????? feature_groups, ????????????essential_feature_groups
        # ----?????????????????????????????????nan, highR_nan, cat, highR_cat num????????????---
        # ?????????????????????????????????(cat????????????)
        for column in X.columns:
            if column not in column2feature_groups:
                # if nan appear, will be consider as nan
                feature_group = self.parse_feature_group(X[column], consider_nan=True)
                # set column2feature_groups
                column2feature_groups[column] = feature_group
                # set column2essential_feature_groups
                if column not in column2essential_feature_groups:
                    if feature_group in NAN_FEATURE_GROUPS:  # ("nan", "highR_nan")
                        essential_feature_group = self.parse_feature_group(X[column], consider_nan=False)
                    else:
                        essential_feature_group = feature_group
                    column2essential_feature_groups[column] = essential_feature_group
        feature_groups = []
        essential_feature_groups = []
        # assemble `feature_groups` , `essential_feature_groups`
        for column in X.columns:
            feature_group = column2feature_groups[column]
            if feature_group not in AUXILIARY_FEATURE_GROUPS:
                feature_groups.append(feature_group)
            essential_feature_group = column2essential_feature_groups[column]
            if essential_feature_group not in AUXILIARY_FEATURE_GROUPS:
                essential_feature_groups.append(essential_feature_group)
        # reindex X_train and X_test
        L1 = self.X_train.shape[0] if self.X_train is not None else 0
        if self.X_test is not None:
            L2 = self.X_test.shape[0]
            self.X_test.index = range(L1, L1 + L2)
        self.X_train.index = range(L1)
        self.feature_groups = feature_groups
        self.column2feature_groups = column2feature_groups
        self.userDefined_column2feature_groups = userDefined_column2feature_groups
        self.essential_feature_groups = essential_feature_groups
        self.column2essential_feature_groups = column2essential_feature_groups
        self.nan_column2essential_fg = self.get_nan_column2essential_fg()
        # todo: ???????????????????????????????????????HDL_Constructor????

    def get_nan_column2essential_fg(self):
        result = {}
        for column, feature_group in self.column2feature_groups.items():
            if feature_group in NAN_FEATURE_GROUPS:
                result[column] = self.column2essential_feature_groups[column]
        return result

    def detect_nan_feature_group(self, series):
        if is_nan(series):
            if is_highR_nan(series, self.highR_nan_threshold):
                return "highR_nan"
            else:
                return "nan"
        return None

    def process_X(self, X: DataFrameContainer, X_origin):
        if X is None:
            return None
        if X.shape[1] != len(self.columns):
            savedpath = os.getenv('SAVEDPATH')
            Path(f"{savedpath}/train_columns.json").write_text(
                json.dumps(self.columns.tolist(), indent=4))
            Path(f"{savedpath}/test_columns.json").write_text(
                json.dumps(X.columns.tolist(), indent=4))
            raise ValueError('feature length dont match')
        if isinstance(X_origin, np.ndarray):
            X.columns = self.columns
        elif isinstance(X_origin, pd.DataFrame):
            # assert set(X.columns) == set(self.columns)
            assert len(X.columns) == len(self.columns), ValueError(
                "model_columns' length should equal to data_columns' length!!!")
            for data_column, model_column in zip(X.columns, self.columns):
                if data_column != model_column:
                    self.logger.warning(
                        f"data_column = {data_column}, model_column = {model_column}, we use data_column instead of model_column. ")
            # fixme: ?????????xenon/data_manager.py:395 ?????????
            # if not np.all(X.columns == self.columns):
            #     self.logger.warning(f"{X.dataset_source}'s columns do not match the TrainSet's columns by position!")
            #     X.data = X.data[self.columns]
        elif isinstance(X_origin, DataFrameContainer):
            pass
        else:
            raise NotImplementedError
        X.set_feature_groups(self.feature_groups)
        return X

    def set_data(self, X_train=None, y_train=None, X_test=None, y_test=None):
        # ??????????????????????????????????????????????????????????????????
        self.X_train, _ = self.parse_data_container("TrainSet", X_train, y_train)
        self.X_test, _ = self.parse_data_container("TestSet", X_test, y_test)
        self.pop_auxiliary_feature_groups()
        self.X_train = self.process_X(self.X_train, X_train)
        self.X_test = self.process_X(self.X_test, X_test)

    def copy(self, keep_data=True):
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        rm = self.resource_manager
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.resource_manager = None
        res = deepcopy(self)
        if keep_data:
            res.X_train = X_train
            res.X_test = X_test
            res.y_train = y_train
            res.y_test = y_test
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.resource_manager = rm
        res.resource_manager = rm
        return res

    def pickle(self, keep_data=True):
        from pickle import dumps
        X_train = self.X_train
        X_test = self.X_test
        y_train = self.y_train
        y_test = self.y_test
        # rm = self.resource_manager
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        # self.resource_manager = None
        res = dumps(self)
        if keep_data:
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test
        # self.resource_manager = rm
        # res.resource_manager = rm
        return res

    def is_empty(self):
        if self.X_train is None and self.X_test is None and self.y_train is None and self.y_test is None:
            return True
        return False
