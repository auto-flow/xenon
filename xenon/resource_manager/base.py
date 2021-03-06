import datetime
import hashlib
import os
import shutil
from copy import deepcopy
from typing import Dict, Tuple, List, Union, Any

import numpy as np
import pandas as pd
import peewee as pw
from frozendict import frozendict
# from playhouse.fields import PickleField
from playhouse.reflection import generate_models

from generic_fs import FileSystem
from generic_fs.utils.db import get_db_class_by_db_type, get_JSONField, create_database
from generic_fs.utils.fs import get_file_system
from xenon.constants import RESOURCE_MANAGER_CLOSE_ALL_LOGGER, ExperimentType
from xenon.data_manager import DataManager
from xenon.ensemble.mean.regressor import MeanRegressor
from xenon.ensemble.vote.classifier import VoteClassifier
from xenon.metrics import Scorer
from xenon.utils.dataframe import replace_dicts, inverse_dict
from xenon.utils.dict_ import update_data_structure, object_kwargs2dict
from xenon.utils.hash import get_hash_of_str, get_hash_of_dict
from xenon.utils.klass import StrSignatureMixin
from xenon.utils.logging_ import get_logger
from xenon.utils.ml_task import MLTask


def get_field_of_type(type_, df, column):
    if not isinstance(type_, str):
        type_ = str(type_)
    type2field = {
        "int64": pw.IntegerField(null=True),
        "int32": pw.IntegerField(null=True),
        "float64": pw.FloatField(null=True),
        "float32": pw.FloatField(null=True),
        "bool": pw.BooleanField(null=True),
    }
    if type_ in type2field:
        return type2field[type_]
    elif type_ == "object":
        try:
            series = df[column]
            N = series.str.len().max()
            if N < 128:
                return pw.CharField(max_length=255, null=True)
            else:
                return pw.TextField(null=True)
        except:
            series = df[column]
            raise NotImplementedError(f"Unsupported type in 'get_field_of_type': '{type(series[0])}'")
    else:
        raise NotImplementedError


class ResourceManager(StrSignatureMixin):
    '''
    ``ResourceManager`` is a class manager computer resources such like ``file_system`` and ``data_base``.
    '''

    def __init__(
            self,
            store_path="~/xenon",
            file_system="local",
            file_system_params=frozendict(),
            db_type="sqlite",
            db_params=frozendict(),
            redis_params=frozendict(),
            max_persistent_estimators=-1,
            compress_suffix="bz2",
            user_id=0,
            search_record_db_name="xenon",
            dataset_table_db_name="xenon_dataset",
            del_local_log_path=True
    ):
        '''

        Parameters
        ----------
        store_path: str
            A path store files, such as metadata and model file and database file, which belong to Xenon.
        file_system: str
            Indicator-string about which file system or storage system will be used.

            Available options list below:
                * ``local``
                * ``hdfs``
                * ``s3``

            ``local`` is default value.
        file_system_params: dict
            Specific file_system configuration.
        db_type: str
            Indicator-string about which file system or storage system will be used.

            Available options list below:
                * ``sqlite``
                * ``postgresql``
                * ``mysql``

            ``sqlite`` is default value.
        db_params: dict
            Specific database configuration.
        redis_params: dict
            Redis configuration.
        max_persistent_estimators: int
            Maximal number of models can persistent in single task.

            If more than this number, the The worst performing model file will be delete,

            the corresponding database record will also be deleted.
        compress_suffix: str
            compress file's suffix, default is bz2
        '''
        self.del_local_log_path = del_local_log_path
        self.dataset_table_db_name = dataset_table_db_name
        self.search_record_db_name = search_record_db_name
        self.user_id = user_id
        # --logger-------------------
        self.logger = get_logger(self)
        self.close_all_logger = get_logger(RESOURCE_MANAGER_CLOSE_ALL_LOGGER)
        # --preprocessing------------
        file_system_params = dict(file_system_params)
        db_params = dict(db_params)
        redis_params = dict(redis_params)
        # ---file_system------------
        self.file_system_type = file_system
        self.file_system: FileSystem = get_file_system(file_system)(**file_system_params)
        if self.file_system_type == "local":
            store_path = os.path.expandvars(os.path.expanduser(store_path))
        self.store_path = store_path
        # ---data_base------------
        assert db_type in ("sqlite", "postgresql", "mysql")
        self.db_type = db_type
        self.db_params = dict(db_params)
        # ---redis----------------
        self.redis_params = dict(redis_params)
        # ---max_persistent_model---
        self.max_persistent_estimators = max_persistent_estimators
        # ---compress_suffix------------
        self.compress_suffix = compress_suffix
        # ---post_process------------
        self.store_path = store_path
        self.file_system.mkdir(self.store_path)
        self.is_init_experiment = False
        self.is_init_task = False
        self.is_init_hdl = False
        self.is_init_trial = False
        self.is_init_dataset = False
        self.is_init_redis = False
        self.is_master = False
        self.is_init_record_db = False
        self.is_init_dataset_db = False
        # --some specific path based on file_system---
        self.datasets_dir = self.file_system.join(self.store_path, "datasets")
        self.databases_dir = self.file_system.join(self.store_path, "databases")
        self.parent_trials_dir = self.file_system.join(self.store_path, "trials")
        self.parent_experiments_dir = self.file_system.join(self.store_path, "experiments")
        for dir_path in [self.datasets_dir, self.databases_dir, self.parent_experiments_dir, self.parent_trials_dir]:
            self.file_system.mkdir(dir_path)
        # --db-----------------------------------------
        self.Datebase = get_db_class_by_db_type(self.db_type)
        # --JSONField-----------------------------------------
        self.JSONField = get_JSONField(self.db_type)
        # --database_name---------------------------------
        # None means didn't create database
        self._dataset_db_name = None
        self._record_db_name = None

    def close_all(self):
        self.close_redis()
        self.close_experiment_table()
        self.close_task_table()
        self.close_hdl_table()
        self.close_trial_table()
        self.close_dataset_table()
        self.close_dataset_db()
        self.close_record_db()
        self.file_system.close_fs()

    def start_safe_close(self):
        pass

    def end_safe_close(self):
        pass

    def __reduce__(self):
        self.close_all()
        return super(ResourceManager, self).__reduce__()

    def update_db_params(self, database):
        db_params = deepcopy(self.db_params)
        if self.db_type == "sqlite":
            db_params["database"] = self.file_system.join(self.databases_dir, f"{database}.db")
        elif self.db_type == "postgresql":
            db_params["database"] = database
        elif self.db_type == "mysql":
            db_params["database"] = database
        else:
            raise NotImplementedError
        return db_params

    def forecast_new_id(self, Dataset, id_field):
        # fixme : ????????????????????????????????????ID????????????????????????
        try:
            records = Dataset.select(getattr(Dataset, id_field)). \
                order_by(-getattr(Dataset, id_field)). \
                limit(1)
            if len(records) == 0:
                estimated_id = 1
            else:
                estimated_id = getattr(records[0], id_field) + 1
        except Exception as e:
            self.logger.error(f"Database Error:\n{e}")
            estimated_id = 1
        return estimated_id

    def persistent_evaluated_model(self, info: Dict, model_id) -> Tuple[str, str, str]:
        y_info = {
            # ???????????????????????????????????????????????????????????????????????????????????????????????????
            "y_true_indexes": info.pop("y_true_indexes"),
            "y_preds": info.pop("y_preds"),
            "y_test_pred": info.pop("y_test_pred", None)
        }
        # ----dir---------------------
        self.trial_dir = self.file_system.join(self.parent_trials_dir, str(self.user_id), self.task_id, self.hdl_id)
        self.file_system.mkdir(self.trial_dir)
        # ----get specific URL---------
        models_path = self.file_system.join(self.trial_dir, f"{model_id}_models.{self.compress_suffix}")
        y_info_path = self.file_system.join(self.trial_dir, f"{model_id}_y-info.{self.compress_suffix}")
        if info.get("finally_fit_model") is not None:
            finally_fit_model_path = self.file_system.join(self.trial_dir,
                                                           f"{model_id}_final.{self.compress_suffix}")
        else:
            finally_fit_model_path = ""
        # ----do dump---------------
        models_path = self.file_system.dump_pickle(info.pop("models"), models_path)
        y_info_path = self.file_system.dump_pickle(y_info, y_info_path)
        if finally_fit_model_path:
            finally_fit_model_path = self.file_system.dump_pickle(info.pop("finally_fit_model"), finally_fit_model_path)
        # ----return----------------
        return models_path, finally_fit_model_path, y_info_path

    def get_ensemble_needed_info(self, task_id) -> Tuple[MLTask, Any]:
        from xenon.data_container import NdArrayContainer

        self.task_id = task_id
        # ??????task?????????trial
        self.init_task_table()
        task_records = self._get_task_records(task_id, self.user_id)
        assert len(task_records) > 0
        task_record = task_records[0]
        ml_task_dict = task_record["ml_task"]
        ml_task = MLTask(**ml_task_dict)
        train_set_id = task_record["train_set_id"]
        test_set_id = task_record["test_set_id"]
        train_label_id = task_record["train_label_id"]
        test_label_id = task_record["test_label_id"]
        y_train = NdArrayContainer(dataset_id=train_label_id, resource_manager=self)
        return ml_task, y_train

    def load_best_estimator(self, ml_task: MLTask):
        self.init_trial_table()
        records = self._get_sorted_trial_records(self.task_id, self.user_id, 1)
        assert len(records) > 0
        record = records[0]
        models = self.file_system.load_pickle(record["models_path"])
        if ml_task.mainTask == "classification":
            estimator = VoteClassifier(models)
        else:
            estimator = MeanRegressor(models)
        return estimator

    def load_best_dhp(self):
        # fixme: ??????hdl_id
        self.init_trial_table()
        trial_id = self._get_best_k_trial_ids(self.task_id, self.user_id, 1)[0]
        record = self._get_trial_records_by_id(trial_id, self.task_id, self.user_id)[0]
        return record["dict_hyper_param"]

    def load_estimators_in_trials(self, trials: Union[List, Tuple], ml_task: MLTask, metric=None) -> Tuple[
        List, List, List, List, List, List]:
        self.init_trial_table()
        records = self._get_trial_records_by_ids(trials, self.task_id, self.user_id)
        estimator_list = []
        y_true_indexes_list = []
        y_preds_list = []
        performance_list = []
        experiment_id_list = []
        scores_list = []  # ???????????????bug???????????????????????????
        if metric is None:
            metric = "mcc" if ml_task.mainTask == "classification" else "r2"

        for record in records:
            exists = True
            if not self.file_system.exists(record["models_path"]):
                exists = False
            else:
                estimator_list.append(self.file_system.load_pickle(record["models_path"]))
            if exists:
                y_info = self.file_system.load_pickle(record["y_info_path"])
                y_true_indexes_list.append(y_info["y_true_indexes"])
                y_preds_list.append(y_info["y_preds"])
                all_score = record["all_score"]
                experiment_id = record["experiment_id"]
                performance = all_score[metric]
                trial_id = record["trial_id"]
                performance_list.append(performance)
                scores_list.append(all_score)
                experiment_id_list.append(experiment_id)
                self.logger.info(f"experiment_id = {experiment_id}\ttrial_id = {trial_id}\t{metric} = {performance}")
        return estimator_list, y_true_indexes_list, y_preds_list, performance_list, scores_list, experiment_id_list

    def set_is_master(self, is_master):
        self.is_master = is_master

    # ----------runhistory------------------------------------------------------------------
    @property
    def runhistory_db_params(self):
        self.init_record_db()
        return self.update_db_params(self.record_db_name)

    def get_runhistory_table_name(self):
        return "run_history"

    @property
    def runhistory_table_name(self):
        return self.get_runhistory_table_name()

    # ----------xenon_dataset------------------------------------------------------------------
    @property
    def dataset_db_name(self):
        if self._dataset_db_name is not None:
            return self._dataset_db_name
        self._dataset_db_name = self.dataset_table_db_name
        create_database(self._dataset_db_name, self.db_type, self.db_params)
        return self._dataset_db_name

    def init_dataset_db(self):
        if self.is_init_dataset_db:
            return self.dataset_db
        else:
            self.is_init_dataset_db = True
            self.dataset_db: pw.Database = self.Datebase(**self.update_db_params(self.dataset_db_name))
            return self.dataset_db

    def close_dataset_db(self):
        self.dataset_db = None
        self.is_init_dataset_db = False

    # ----------xenon------------------------------------------------------------------

    @property
    def record_db_name(self):
        if self._record_db_name is not None:
            return self._record_db_name
        self._record_db_name = self.search_record_db_name
        create_database(self._record_db_name, self.db_type, self.db_params)
        return self._record_db_name

    def init_record_db(self):
        if self.is_init_record_db:
            return self.record_db
        else:
            self.is_init_record_db = True
            self.record_db: pw.Database = self.Datebase(**self.update_db_params(self.record_db_name))
            return self.record_db

    def close_record_db(self):
        self.record_db = None
        self.is_init_record_db = False

    # ----------redis------------------------------------------------------------------

    def connect_redis(self):
        if self.is_init_redis:
            return True
        try:
            from redis import Redis
            self.redis_client = Redis(**self.redis_params)
            self.is_init_redis = True
            return True
        except Exception as e:
            self.logger.error(f"Redis Error:\n{e}")
            return False

    def close_redis(self):
        self.redis_client = None
        self.is_init_redis = False

    def clear_pid_list(self):
        self.redis_delete("pid_list")

    def push_pid_list(self):
        if self.connect_redis():
            self.redis_client.rpush("pid_list", os.getpid())

    def get_pid_list(self):
        if self.connect_redis():
            l = self.redis_client.lrange("pid_list", 0, -1)
            return list(map(lambda x: int(x.decode()), l))
        else:
            return []

    def redis_set(self, name, value, ex=None, px=None, nx=False, xx=False):
        if self.connect_redis():
            self.redis_client.set(name, value, ex, px, nx, xx)

    def redis_get(self, name):
        if self.connect_redis():
            return self.redis_client.get(name)
        else:
            return None

    def redis_hset(self, name, key, value):
        if self.connect_redis():
            try:
                self.redis_client.hset(name, key, value)
            except Exception as e:
                pass

    def redis_hgetall(self, name):
        if self.connect_redis():
            return self.redis_client.hgetall(name)
        else:
            return None

    def redis_delete(self, name):
        if self.connect_redis():
            self.redis_client.delete(name)

    # ----------dataset_model------------------------------------------------------------------
    def get_dataset_model(self) -> pw.Model:
        class Dataset(pw.Model):
            dataset_id = pw.FixedCharField(max_length=32)
            user_id = pw.IntegerField()
            dataset_metadata = self.JSONField(default={})
            dataset_path = pw.TextField(default="")
            upload_type = pw.CharField(max_length=32)
            dataset_type = pw.CharField(max_length=32)
            dataset_source = pw.CharField(max_length=32)
            column_descriptions = self.JSONField(default={})
            columns_mapper = self.JSONField(default={})
            columns = self.JSONField(default={})
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.record_db
                primary_key = pw.CompositeKey('dataset_id', 'user_id')

        self.record_db.create_tables([Dataset])
        return Dataset

    def get_dataset_path(self, dataset_id):
        dataset_dir = self.file_system.join(self.datasets_dir, str(self.user_id))
        self.file_system.mkdir(dataset_dir)
        dataset_path = self.file_system.join(dataset_dir, f"{dataset_id}.h5")
        return dataset_path

    def insert_dataset_record(
            self,
            dataset_id,
            dataset_metadata,
            dataset_type,
            dataset_path,
            upload_type,
            dataset_source,
            column_descriptions,
            columns_mapper,
            columns
    ):
        self.init_dataset_table()
        return self._insert_dataset_record(
            self.user_id,
            dataset_id,
            dataset_metadata,
            dataset_type,
            dataset_path,
            upload_type,
            dataset_source,
            column_descriptions,
            columns_mapper,
            columns
        )

    def _insert_dataset_record(
            self,
            user_id: int,
            dataset_id: str,
            dataset_metadata: Dict[str, Any],
            dataset_type: str,
            dataset_path: str,
            upload_type: str,
            dataset_source: str,
            column_descriptions: Dict[str, Any],
            columns_mapper: Dict[str, str],
            columns: List[str]
    ):
        records = self.DatasetModel.select().where(
            (self.DatasetModel.dataset_id == dataset_id) & (self.DatasetModel.user_id == user_id)
        )
        L = len(records)
        if L != 0:
            record = records[0]
            record.modify_time = datetime.datetime.now()
            record.dataset_metadata = dataset_metadata
            record.save()
        else:
            record = self.DatasetModel().create(
                dataset_id=dataset_id,
                user_id=self.user_id,
                dataset_metadata=dataset_metadata,
                dataset_path=dataset_path,
                dataset_type=dataset_type,
                upload_type=upload_type,
                dataset_source=dataset_source,
                column_descriptions=column_descriptions,
                columns_mapper=columns_mapper,
                columns=columns
            )
        dataset_id = record.dataset_id
        return {
            "length": L,
            "dataset_id": dataset_id,
            "dataset_path": dataset_path,
        }

    def init_dataset_table(self):
        if self.is_init_dataset:
            return
        self.is_init_dataset = True
        self.init_record_db()
        self.DatasetModel = self.get_dataset_model()

    def close_dataset_table(self):
        self.is_init_dataset = False
        self.DatasetModel = None

    def upload_df_to_fs(self, df: pd.DataFrame, dataset_path):
        tmp_path = f"/tmp/tmp_df_{os.getpid()}.h5"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        df.to_hdf(tmp_path, "dataset")
        return self.file_system.upload(dataset_path, tmp_path)

    def upload_ndarray_to_fs(self, arr: np.ndarray, dataset_path):
        import h5py
        tmp_path = f"/tmp/tmp_arr_{os.getpid()}.h5"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        with h5py.File(tmp_path, 'w') as hf:
            hf.create_dataset("dataset", data=arr)
        return self.file_system.upload(dataset_path, tmp_path)

    def get_dataset_records(self, dataset_id) -> List[Dict[str, Any]]:
        self.init_dataset_table()
        return self._get_dataset_records(dataset_id, self.user_id)

    def _get_dataset_records(self, dataset_id, user_id) -> List[Dict[str, Any]]:
        self.init_dataset_table()
        records = self.DatasetModel.select().where(
            (self.DatasetModel.dataset_id == dataset_id) & (self.DatasetModel.user_id == user_id)
        ).dicts()
        return list(records)

    def download_df_from_table(self, dataset_id, columns, columns_mapper):
        inv_columns_mapper = inverse_dict(columns_mapper)
        dataset_db = self.init_dataset_db()
        models = generate_models(dataset_db)
        table_name = f"dataset_{dataset_id}"
        if table_name not in models:
            raise ValueError(f"Table {table_name} didn't exists.")
        model = models[table_name]
        L = 500
        offset = 0
        dataframes = []
        while True:
            dicts = list(model().select().limit(L).offset(offset).dicts())
            replace_dicts(dicts, None, np.nan)
            sub_df = pd.DataFrame(dicts)
            dataframes.append(sub_df)
            if len(dicts) < L:
                break
            offset += L
        df = pd.concat(dataframes, axis=0)
        df.index = range(df.shape[0])
        # ?????????????????????????????????
        database_id = df.columns[0]
        df.pop(database_id)
        df.columns = df.columns.map(inv_columns_mapper)
        if columns is not None:
            df = df[columns]
        return df

    def download_df_from_fs(self, dataset_path):
        tmp_path = f"/tmp/tmp_df_{get_hash_of_str(dataset_path)}.h5"
        self.file_system.download(dataset_path, tmp_path)
        df: pd.DataFrame = pd.read_hdf(tmp_path, "dataset")
        return df

    def download_arr_from_fs(self, dataset_path):
        tmp_path = f"/tmp/tmp_arr_{get_hash_of_str(dataset_path)}.h5"
        self.file_system.download(dataset_path, tmp_path)
        import h5py
        with h5py.File(tmp_path, 'r') as hf:
            arr = hf['dataset'][:]
        return arr

    # ----------experiment_model------------------------------------------------------------------
    def get_experiment_model(self) -> pw.Model:
        class Experiment(pw.Model):
            experiment_id = pw.AutoField(primary_key=True)
            user_id = pw.IntegerField()
            hdl_id = pw.FixedCharField(max_length=32, null=True)
            task_id = pw.FixedCharField(max_length=32)
            experiment_type = pw.CharField(max_length=128)  # auto_modeling, manual_modeling, ensemble_modeling
            experiment_config = self.JSONField(default={})  # ?????????????????????????????????????????????????????????
            additional_info = self.JSONField(default={})  # trials???experiments????????????
            final_model_path = pw.TextField(null=True)
            log_path = pw.TextField(null=True)
            start_time = pw.DateTimeField(default=datetime.datetime.now)
            end_time = pw.DateTimeField(null=True)

            class Meta:
                database = self.record_db

        self.record_db.create_tables([Experiment])
        return Experiment

    def insert_experiment_record(
            self,
            experiment_type: ExperimentType,
            experiment_config,
            additional_info,
    ):
        self.init_experiment_table()
        assert isinstance(experiment_type, ExperimentType)
        self.experiment_id = self._insert_experiment_record(self.user_id, getattr(self, "hdl_id", None), self.task_id,
                                                            experiment_type.value,
                                                            experiment_config, additional_info)

    def _insert_experiment_record(
            self, user_id: int, hdl_id: str, task_id: str,
            experiment_type: str,
            experiment_config: Dict[str, Any], additional_info: Dict[str, Any]
    ):
        self.additional_info = additional_info
        experiment_record = self.ExperimentModel.create(
            user_id=user_id,
            hdl_id=hdl_id,
            task_id=task_id,
            experiment_type=experiment_type,
            experiment_config=experiment_config,
            additional_info=additional_info,
        )
        return experiment_record.experiment_id

    def finish_experiment(self, local_log_path, final_model, del_local_log_path=None):
        if del_local_log_path is None:
            del_local_log_path = self.del_local_log_path
        # ????????????
        self.experiment_path = self.file_system.join(self.parent_experiments_dir, str(self.user_id),
                                                     str(self.experiment_id))
        self.file_system.mkdir(self.experiment_path)
        experiment_log_path = self.file_system.join(self.experiment_path, "log_file.log")
        experiment_model_path = self.file_system.join(self.experiment_path, "model.bz2")
        # ???????????????????????????
        final_model = final_model.copy()
        assert final_model.data_manager.is_empty()
        self.start_safe_close()
        experiment_model_path = self.file_system.dump_pickle(final_model, experiment_model_path)
        self.end_safe_close()
        # ????????????
        if os.path.exists(local_log_path):
            tmp_log_path = f"/tmp/log"
            if os.path.exists(tmp_log_path):
                os.remove(tmp_log_path)
            shutil.copy(local_log_path, tmp_log_path)
            if del_local_log_path:
                os.remove(local_log_path)
            experiment_log_path = self.file_system.upload(experiment_log_path, tmp_log_path)
        else:
            experiment_log_path = ""
            self.logger.warning(f"Local log path : '{local_log_path}' didn't exist!")
        # ?????????????????????
        self.finish_experiment_update_info(experiment_model_path, experiment_log_path, datetime.datetime.now())

    def finish_experiment_update_info(self, final_model_path, log_path, end_time):
        self.init_experiment_table()
        self._finish_experiment_update_info(self.experiment_id, final_model_path, log_path, end_time)

    def _finish_experiment_update_info(self, experiment_id: int, final_model_path: str, log_path: str,
                                       end_time: Union[datetime.datetime, str]):
        experiment = self.ExperimentModel.select().where(self.ExperimentModel.experiment_id == experiment_id)[0]
        experiment.final_model_path = final_model_path
        experiment.log_path = log_path
        experiment.end_time = end_time
        experiment.save()

    def _get_experiment_record(self, experiment_id):
        experiment_records = self.ExperimentModel.select().where(
            self.ExperimentModel.experiment_id == experiment_id).dicts()
        return list(experiment_records)

    def init_experiment_table(self):
        if self.is_init_experiment:
            return
        self.is_init_experiment = True
        self.init_record_db()
        self.ExperimentModel = self.get_experiment_model()

    def close_experiment_table(self):
        self.is_init_experiment = False
        self.ExperimentModel = None

    # ----------task_model------------------------------------------------------------------
    def get_task_model(self) -> pw.Model:
        class Task(pw.Model):
            task_id = pw.FixedCharField(max_length=32)
            user_id = pw.IntegerField()
            metric = pw.CharField(max_length=256)
            splitter = self.JSONField()  # pw.TextField()
            ml_task = self.JSONField()  # pw.CharField(max_length=256)
            specific_task_token = pw.CharField(max_length=256, default="")
            train_set_id = pw.FixedCharField(max_length=32)
            test_set_id = pw.FixedCharField(max_length=32, default="")
            train_label_id = pw.FixedCharField(max_length=32)
            test_label_id = pw.FixedCharField(max_length=32, default="")
            sub_sample_indexes = self.JSONField(default=[])
            sub_feature_indexes = self.JSONField(default=[])
            task_metadata = self.JSONField(default={})
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.record_db
                primary_key = pw.CompositeKey('task_id', 'user_id')

        self.record_db.create_tables([Task])
        return Task

    def insert_task_record(self, data_manager: DataManager,
                           metric: Scorer, splitter,
                           specific_task_token, dataset_metadata,
                           task_metadata, sub_sample_indexes, sub_feature_indexes):
        self.init_task_table()
        train_set_id = data_manager.train_set_id
        test_set_id = data_manager.test_set_id
        train_label_id = data_manager.train_label_id
        test_label_id = data_manager.test_label_id
        metric_str = metric.name
        splitter_str = str(splitter)
        splitter_dict = object_kwargs2dict(splitter, contain_class_name=True)
        ml_task = data_manager.ml_task
        ml_task_str = str(ml_task)
        ml_task_dict = object_kwargs2dict(ml_task, func="__new__", keys=ml_task._fields)
        if sub_sample_indexes is None:
            sub_sample_indexes = []
        if sub_feature_indexes is None:
            sub_feature_indexes = []
        if not isinstance(sub_sample_indexes, list):
            sub_sample_indexes = list(sub_sample_indexes)
        if not isinstance(sub_feature_indexes, list):
            sub_feature_indexes = list(sub_feature_indexes)
        sub_sample_indexes_str = str(sub_sample_indexes)
        sub_feature_indexes_str = str(sub_feature_indexes)
        # ---task_id----------------------------------------------------
        m = hashlib.md5()
        get_hash_of_str(train_set_id, m)
        get_hash_of_str(test_set_id, m)
        get_hash_of_str(train_label_id, m)
        get_hash_of_str(test_label_id, m)
        get_hash_of_str(metric_str, m)
        get_hash_of_str(splitter_str, m)
        get_hash_of_str(ml_task_str, m)
        get_hash_of_str(sub_sample_indexes_str, m)
        get_hash_of_str(sub_feature_indexes_str, m)
        get_hash_of_str(specific_task_token, m)
        task_hash = m.hexdigest()
        task_id = task_hash
        task_metadata = dict(
            dataset_metadata=dataset_metadata, **task_metadata
        )
        self.task_id = task_id
        self._insert_task_record(
            task_id, self.user_id, metric_str, splitter_dict, ml_task_dict, train_set_id,
            test_set_id, train_label_id, test_label_id, specific_task_token, task_metadata,
            sub_sample_indexes, sub_feature_indexes
        )

    def _insert_task_record(self, task_id: str, user_id: int,
                            metric_str: str, splitter_dict: Dict[str, str], ml_task_dict: Dict[str, str],
                            train_set_id: str, test_set_id: str, train_label_id: str, test_label_id: str,
                            specific_task_token: str, task_metadata: Dict[str, Any], sub_sample_indexes: List[int],
                            sub_feature_indexes: List[str]):
        records = self.TaskModel.select().where(
            (self.TaskModel.task_id == task_id) & (self.TaskModel.user_id == user_id)
        )
        if len(records) == 0:
            self.TaskModel.create(
                task_id=task_id,
                user_id=user_id,
                metric=metric_str,
                splitter=splitter_dict,
                ml_task=ml_task_dict,
                specific_task_token=specific_task_token,
                train_set_id=train_set_id,
                test_set_id=test_set_id,
                train_label_id=train_label_id,
                test_label_id=test_label_id,
                sub_sample_indexes=sub_sample_indexes,
                sub_feature_indexes=sub_feature_indexes,
                task_metadata=task_metadata
            )
        else:
            record = records[0]
            old_meta_data = record.task_metadata
            new_meta_data = update_data_structure(old_meta_data, task_metadata)
            record.task_metadata = new_meta_data
            record.save()
        return task_id

    def _get_task_records(self, task_id: str, user_id: int):
        task_records = self.TaskModel.select().where(
            (self.TaskModel.task_id == task_id) & (self.TaskModel.user_id == user_id)
        ).dicts()
        task_records = list(task_records)
        return task_records

    def init_task_table(self):
        if self.is_init_task:
            return
        self.is_init_task = True
        self.init_record_db()
        self.TaskModel = self.get_task_model()

    def close_task_table(self):
        self.is_init_task = False
        self.TaskModel = None

    # ----------hdl_model------------------------------------------------------------------
    def get_hdl_model(self) -> pw.Model:
        class Hdl(pw.Model):
            task_id = pw.FixedCharField(max_length=32)
            hdl_id = pw.FixedCharField(max_length=32)
            user_id = pw.IntegerField()
            hdl = self.JSONField(default={})
            hdl_metadata = self.JSONField(default={})
            create_time = pw.DateTimeField(default=datetime.datetime.now)
            modify_time = pw.DateTimeField(default=datetime.datetime.now)

            class Meta:
                database = self.record_db
                primary_key = pw.CompositeKey('task_id', 'hdl_id', 'user_id')

        self.record_db.create_tables([Hdl])
        return Hdl

    def insert_hdl_record(self, hdl, hdl_metadata):
        self.init_hdl_table()
        hdl_hash = get_hash_of_dict(hdl)
        hdl_id = hdl_hash
        self.hdl_id = hdl_id
        self._insert_hdl_record(self.task_id, hdl_id, self.user_id, hdl, hdl_metadata)

    def _insert_hdl_record(self, task_id: str, hdl_id: str, user_id: int, hdl: dict, hdl_metadata: Dict[str, Any]):
        records = self.HdlModel.select().where(
            (self.HdlModel.task_id == task_id) & (self.HdlModel.hdl_id == hdl_id))
        if len(records) == 0:
            self.HdlModel.create(
                task_id=task_id,
                hdl_id=hdl_id,
                user_id=user_id,
                hdl=hdl,
                hdl_metadata=hdl_metadata
            )
        else:
            record = records[0]
            old_meta_data = record.hdl_metadata
            new_meta_data = update_data_structure(old_meta_data, hdl_metadata)
            record.hdl_metadata = new_meta_data
            record.save()
        return hdl_id

    def init_hdl_table(self):
        if self.is_init_hdl:
            return
        self.is_init_hdl = True
        self.init_record_db()
        self.HdlModel = self.get_hdl_model()

    def close_hdl_table(self):
        self.is_init_hdl = False
        self.HdlModel = None

    # ----------trial_model------------------------------------------------------------------

    def get_trial_model(self) -> pw.Model:
        class Trial(pw.Model):
            trial_id = pw.AutoField(primary_key=True)
            user_id = pw.IntegerField()
            config_id = pw.FixedCharField(max_length=32)
            run_id = pw.FixedCharField(max_length=256)
            instance_id = pw.FixedCharField(max_length=128)
            experiment_id = pw.IntegerField()
            task_id = pw.FixedCharField(max_length=32, index=True)  # ?????????
            hdl_id = pw.FixedCharField(max_length=32)
            estimator = pw.CharField(max_length=256, default="")
            loss = pw.FloatField(default=65535)
            losses = self.JSONField(default=[])
            test_loss = self.JSONField(default=[])  # ?????????
            all_score = self.JSONField(default={})
            all_scores = self.JSONField(default=[])
            test_all_score = self.JSONField(default={})  # ?????????
            models_path = pw.TextField(default="")
            final_model_path = pw.TextField(default="")
            y_info_path = pw.TextField(default="")
            additional_info = self.JSONField(default={})
            # smac_hyper_param = PickleField(default=0)
            dict_hyper_param = self.JSONField(default={})
            cost_time = pw.FloatField(default=65535)
            status = pw.CharField(max_length=32, default="SUCCESS")
            failed_info = pw.TextField(default="")
            warning_info = pw.TextField(default="")
            intermediate_results = self.JSONField(default=[])
            start_time = pw.DateTimeField()
            end_time = pw.DateTimeField()

            class Meta:
                database = self.record_db

        self.record_db.create_tables([Trial])
        return Trial

    def init_trial_table(self):
        if self.is_init_trial:
            return
        self.is_init_trial = True
        self.init_record_db()
        self.TrialsModel = self.get_trial_model()

    def close_trial_table(self):
        self.is_init_trial = False
        self.TrialsModel = None

    def insert_trial_record(self, info: Dict):
        self.init_trial_table()
        config_id = info.get("config_id")
        models_path, finally_fit_model_path, y_info_path = \
            self.persistent_evaluated_model(info, config_id)
        info.update(
            models_path=models_path,
            finally_fit_model_path=finally_fit_model_path,
            y_info_path=y_info_path,
        )
        return self._insert_trial_record(self.user_id, self.task_id, self.hdl_id, self.experiment_id, info)

    def _get_sorted_trial_records(self, task_id, user_id, limit):
        records = self.TrialsModel.select().where(
            (self.TrialsModel.task_id == task_id) & (self.TrialsModel.user_id == user_id)
        ).order_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(limit).dicts()
        return list(records)

    def _get_trial_records_by_id(self, trial_id, task_id, user_id):
        records = self.TrialsModel.select().where(
            (self.TrialsModel.trial_id == trial_id) & (self.TrialsModel.task_id == task_id) &
            (self.TrialsModel.user_id == user_id)
        ).dicts()
        return list(records)

    def _get_trial_records_by_ids(self, trial_ids, task_id, user_id):
        records = self.TrialsModel.select().where(
            (self.TrialsModel.trial_id << trial_ids) & (self.TrialsModel.task_id == task_id) &
            (self.TrialsModel.user_id == user_id)
        ).dicts()
        records = list(records)
        return records

    def _get_best_k_trial_ids(self, task_id, user_id, k):
        # self.init_trial_table()
        trial_ids = []
        records = self.TrialsModel.select(self.TrialsModel.trial_id).where(
            (self.TrialsModel.task_id == task_id) & (self.TrialsModel.user_id == user_id)
        ).order_by(self.TrialsModel.loss, self.TrialsModel.cost_time).limit(k)
        for record in records:
            trial_ids.append(record.trial_id)
        return trial_ids

    def _insert_trial_record(self, user_id: int, task_id: str, hdl_id: str, experiment_id: int, info: dict):
        success = False
        max_try_times = 3
        trial_id = -1
        for i in range(max_try_times):
            try:
                trial_id = self.do_insert_trial_record(user_id, task_id, hdl_id, experiment_id, info)
                success = True
            except Exception as e:
                self.logger.error(e)
                self.logger.error(f"Insert 'trial' table failed, {i + 1} try.")
                # ?????????????????? ????????????
                self.close_trial_table()
                self.close_record_db()
                self.init_record_db()
                self.init_trial_table()
            if success:
                break
        if not success:
            self.logger.error(f"After {max_try_times} times try, trial info cannot insert into trial table.")
        return trial_id

    def do_insert_trial_record(self, user_id, task_id, hdl_id, experiment_id, info: dict):
        trial_record = self.TrialsModel.create(
            user_id=user_id,
            config_id=info.get("config_id"),
            run_id=info.get("run_id"),
            instance_id=info.get("instance_id"),
            task_id=task_id,
            hdl_id=hdl_id,
            experiment_id=experiment_id,
            estimator=info.get("component", ""),
            loss=info.get("loss", 65535),
            losses=info.get("losses", []),
            test_loss=info.get("test_loss", 65535),
            all_score=info.get("all_score", {}),
            all_scores=info.get("all_scores", []),
            test_all_score=info.get("test_all_score", {}),
            models_path=info.get("models_path", ""),
            final_model_path=info.get("finally_fit_model_path", ""),
            y_info_path=info.get("y_info_path", ""),
            additional_info=info.get("additional_info", {}),
            # smac_hyper_param=info.get("program_hyper_param"),
            dict_hyper_param=info.get("dict_hyper_param", {}),
            cost_time=info.get("cost_time", 65535),
            status=info.get("status", "failed"),
            failed_info=info.get("failed_info", ""),
            warning_info=info.get("warning_info", ""),
            intermediate_results=info.get("intermediate_results", []),
            start_time=info.get("start_time"),
            end_time=info.get("end_time"),
        )
        return trial_record.trial_id

    def delete_models(self):
        # if hasattr(self, "sync_dict"):
        #     exit_processes = self.sync_dict.get("exit_processes", 3)
        #     records = 0
        #     for key, value in self.sync_dict.items():
        #         if isinstance(key, int):
        #             records += value
        #     if records >= exit_processes:
        #         return False
        # master segment
        if not self.is_master:
            return True
        self.init_trial_table()
        # todo: ??????????????????
        if self.max_persistent_estimators > 0:
            # ???????????????task & hdl????????????????????????
            should_delete = self.TrialsModel.select().where(
                (self.TrialsModel.task_id == self.task_id) & (self.TrialsModel.user_id == self.user_id)
                & (self.TrialsModel.hdl_id == self.hdl_id)
            ).order_by(
                self.TrialsModel.loss, self.TrialsModel.cost_time
            ).offset(self.max_persistent_estimators)
            if len(should_delete):
                for record in should_delete:
                    models_path = record.models_path
                    self.logger.info(f"Delete expire Model in path : {models_path}")
                    self.file_system.delete(models_path)
                self.TrialsModel.delete().where(
                    self.TrialsModel.trial_id.in_(should_delete.select(self.TrialsModel.trial_id))).execute()
        return True
