from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["GradientBoostingRegressor"]


class GradientBoostingRegressor(XenonIterComponent, XenonRegressionAlgorithm):
    class__ = "GradientBoostingRegressor"
    module__ = "sklearn.ensemble"
