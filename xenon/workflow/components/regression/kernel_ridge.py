from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["KernelRidge"]


class KernelRidge(XenonRegressionAlgorithm):
    class__ = "KernelRidge"
    module__ = "sklearn.kernel_ridge"
