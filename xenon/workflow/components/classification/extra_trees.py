from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["ExtraTreesClassifier"]


class ExtraTreesClassifier(
    XenonIterComponent, XenonClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
    tree_model = True
