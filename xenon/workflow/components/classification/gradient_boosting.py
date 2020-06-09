from copy import deepcopy

from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["GradientBoostingClassifier"]


class GradientBoostingClassifier(XenonIterComponent, XenonClassificationAlgorithm):
    module__ =  "sklearn.ensemble"
    class__ = "GradientBoostingClassifier"


