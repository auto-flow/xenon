from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["TruncatedSVD"]

class TruncatedSVD(XenonFeatureEngineerAlgorithm):
    class__ = "TruncatedSVD"
    module__ = "sklearn.decomposition"