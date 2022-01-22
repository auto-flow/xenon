from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["RandomForestRegressor"]


class RandomForestRegressor(
    XenonIterComponent, XenonRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"
