from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["FastICA"]

class FastICA(XenonFeatureEngineerAlgorithm):
    class__ = "FastICA"
    module__ = "sklearn.decomposition"
