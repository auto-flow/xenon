from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["DropAll"]


class DropAll(XenonFeatureEngineerAlgorithm):
    class__ = "DropAll"
    module__ = "xenon.feature_engineer.operate.drop_all"
