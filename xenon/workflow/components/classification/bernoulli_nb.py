import numpy as np

from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__=["BernoulliNB"]


class BernoulliNB(XenonClassificationAlgorithm):
    class__ = "BernoulliNB"
    module__ = "sklearn.naive_bayes"
    OVR__ = True
