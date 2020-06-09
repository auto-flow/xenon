from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["ElasticNet"]

class ElasticNet(XenonIterComponent, XenonRegressionAlgorithm):
    class__ = "ElasticNet"
    module__ = "sklearn.linear_model"
    iterations_name = "max_iter"

