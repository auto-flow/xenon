from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["QDA"]


class QDA(XenonClassificationAlgorithm):
    class__ = "QuadraticDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True
