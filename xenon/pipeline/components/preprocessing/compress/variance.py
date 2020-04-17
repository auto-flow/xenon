from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["Variance"]

class Variance(XenonFeatureEngineerAlgorithm):
    class__ = "Variance"
    module__ = "xenon.feature_engineer.compress.variance"