
from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class DecisionTree(XenonRegressionAlgorithm):
    module__ = "sklearn.tree"
    class__ = "DecisionTreeRegressor"
