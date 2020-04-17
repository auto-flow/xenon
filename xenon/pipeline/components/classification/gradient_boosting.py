from copy import deepcopy

from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm

__all__=["GradientBoostingClassifier"]


class GradientBoostingClassifier(XenonClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"


