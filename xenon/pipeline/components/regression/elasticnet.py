from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class ElasticNet(XenonRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"

