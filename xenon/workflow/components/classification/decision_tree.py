from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["DecisionTree"]


class DecisionTree(XenonClassificationAlgorithm):
    class__ = "DecisionTreeClassifier"
    module__ = "sklearn.tree"
