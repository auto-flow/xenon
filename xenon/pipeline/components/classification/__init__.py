import os

from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm
from xenon.utils.packages import find_components

classifier_directory = os.path.split(__file__)[0]
_classifiers = find_components(__package__,
                               classifier_directory,
                               XenonClassificationAlgorithm)
