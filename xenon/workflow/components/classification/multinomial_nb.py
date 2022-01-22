from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["MultinomialNB"]

class MultinomialNB(XenonClassificationAlgorithm):
    module__ = "sklearn.naive_bayes"
    class__ = "MultinomialNB"
    OVR__ = True

