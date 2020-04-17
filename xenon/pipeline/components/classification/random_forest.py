from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm

__all__=["RandomForest"]

class RandomForest(XenonClassificationAlgorithm):
    class__ = "RandomForestClassifier"
    module__ = "sklearn.ensemble"