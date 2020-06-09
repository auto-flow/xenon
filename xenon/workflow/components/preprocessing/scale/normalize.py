from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["NormalizerComponent"]

class NormalizerComponent(XenonFeatureEngineerAlgorithm):
    class__ = "Normalizer"
    module__ = "sklearn.preprocessing"