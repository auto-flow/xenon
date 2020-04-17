from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class ARDRegression(XenonRegressionAlgorithm):
    class__ = "ARDRegression"
    module__ = "sklearn.linear_model"

