from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["LogisticRegression"]

class LogisticRegression(XenonIterComponent, XenonClassificationAlgorithm):
    class__ = "LogisticRegression"
    module__ = "sklearn.linear_model"
    iterations_name = "max_iter"
