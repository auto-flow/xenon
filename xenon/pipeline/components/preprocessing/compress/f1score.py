
from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["F1Score"]


class F1Score(XenonFeatureEngineerAlgorithm):
    class__ = "F1Score"
    module__ = "xenon.feature_engineer.compress.f1score"
    store_intermediate = True
    suspend_other_processes = True