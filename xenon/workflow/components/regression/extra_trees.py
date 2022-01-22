from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["ExtraTreesRegressor"]


class ExtraTreesRegressor(
    XenonIterComponent, XenonRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ExtraTreesRegressor"
