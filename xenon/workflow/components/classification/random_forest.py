from typing import Dict

from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["RandomForestClassifier"]


class RandomForestClassifier(XenonIterComponent, XenonClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"


