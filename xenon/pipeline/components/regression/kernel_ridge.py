from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class ElasticNet(XenonRegressionAlgorithm):
    class__ = "KernelRidge"
    module__ = "sklearn.kernel_ridge"

