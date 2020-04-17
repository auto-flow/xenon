from copy import deepcopy
from typing import Dict

from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm

__all__=["LDA"]

class LDA(XenonClassificationAlgorithm):
    class__ = "LinearDiscriminantAnalysis"
    module__ = "sklearn.discriminant_analysis"
    OVR__ = True


