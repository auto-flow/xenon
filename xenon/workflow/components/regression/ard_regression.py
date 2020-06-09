from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["ARDRegression"]


class ARDRegression(XenonRegressionAlgorithm):
    class__ = "ARDRegression"
    module__ = "sklearn.linear_model"
