
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["DecisionTree"]

class DecisionTree(XenonRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
