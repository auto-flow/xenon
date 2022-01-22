from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["LibSVM_SVC"]


class LibSVM_SVC(XenonClassificationAlgorithm):
    class__ = "SVC"
    module__ = "sklearn.svm"
