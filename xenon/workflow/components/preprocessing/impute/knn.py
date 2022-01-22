from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["KNNImputer"]


class KNNImputer(XenonFeatureEngineerAlgorithm):
    class__ = "KNNImputer"
    module__ = "xenon.feature_engineer.impute"
