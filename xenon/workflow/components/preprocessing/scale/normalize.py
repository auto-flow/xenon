from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["Normalizer"]

class Normalizer(XenonFeatureEngineerAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"