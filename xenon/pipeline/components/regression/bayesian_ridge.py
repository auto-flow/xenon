from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class BayesianRidge(XenonRegressionAlgorithm):
    class__ = "BayesianRidge"
    module__ = "sklearn.linear_model"

