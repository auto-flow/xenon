from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["QuantileTransformer"]


class QuantileTransformer(XenonFeatureEngineerAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"
