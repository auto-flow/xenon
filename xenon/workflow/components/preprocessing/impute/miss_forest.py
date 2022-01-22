from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["MissForest"]


class MissForest(XenonFeatureEngineerAlgorithm):
    class__ = "MissForest"
    module__ = "xenon.feature_engineer.impute"
