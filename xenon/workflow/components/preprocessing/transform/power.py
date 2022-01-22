from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["PowerTransformer"]


class PowerTransformer(XenonFeatureEngineerAlgorithm):
    class__ = "PowerTransformer"
    module__ = "sklearn.preprocessing"
