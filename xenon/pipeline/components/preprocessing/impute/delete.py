from functools import partial
from typing import Dict

import pandas as pd
from imblearn import FunctionSampler
from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm
from xenon.pipeline.dataframe import GenericDataFrame

__all__ = ["DeleteNanRow"]

class DeleteNanRow(XenonDataProcessAlgorithm):
    class__ = "DeleteNanRow"
    module__ = "xenon.data_process.impute.delete_nan"
