from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm

__all__=["ExtraTreesClassifier"]


class ExtraTreesClassifier(
    XenonClassificationAlgorithm,
):
    class__ = "ExtraTreesClassifier"
    module__ = "sklearn.ensemble"
    tree_model = True
