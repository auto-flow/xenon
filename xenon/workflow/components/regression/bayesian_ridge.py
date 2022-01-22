from xenon.workflow.components.regression_base import XenonRegressionAlgorithm


__all__ = ["BayesianRidge"]


class BayesianRidge(XenonRegressionAlgorithm):
    class__ = "BayesianRidge"
    module__ = "sklearn.linear_model"

