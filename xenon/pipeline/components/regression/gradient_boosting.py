from copy import deepcopy

from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class GradientBoosting(XenonRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"


