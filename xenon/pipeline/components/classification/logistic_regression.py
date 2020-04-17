from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm

__all__=["LogisticRegression"]

class LogisticRegression(XenonClassificationAlgorithm):
    class__ = "LogisticRegression"
    module__ = "sklearn.linear_model"
