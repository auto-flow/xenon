from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["LibSVM_SVR"]


class LibSVM_SVR(XenonRegressionAlgorithm):
    class__ = "SVR"
    module__ = "sklearn.svm"
