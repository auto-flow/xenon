from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["PCA"]

class PCA(XenonFeatureEngineerAlgorithm):
    class__ = "PCA"
    module__ = "sklearn.decomposition"



