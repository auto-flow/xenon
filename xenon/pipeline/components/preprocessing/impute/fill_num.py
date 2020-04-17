from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["FillNum"]


class FillNum(XenonFeatureEngineerAlgorithm):
    class__ = "SimpleImputer"
    module__ = "sklearn.impute"
