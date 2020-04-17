from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["StandardScaler"]

class StandardScaler(XenonFeatureEngineerAlgorithm):
    class__ = "StandardScaler"
    module__ = "sklearn.preprocessing"
