from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["Pearson"]

class Pearson(XenonFeatureEngineerAlgorithm):
    class__ = "Pearson"
    module__ = "xenon.feature_engineer.compress.pearson"
    cache_intermediate = True
    suspend_other_processes = True