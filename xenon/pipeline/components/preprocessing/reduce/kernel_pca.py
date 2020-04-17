from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["KernelPCA"]

class KernelPCA(XenonFeatureEngineerAlgorithm):
    class__ = "KernelPCA"
    module__ = "sklearn.decomposition"

