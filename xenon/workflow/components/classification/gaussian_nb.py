from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["GaussianNB"]


class GaussianNB(XenonClassificationAlgorithm):
    class__ = "GaussianNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True

