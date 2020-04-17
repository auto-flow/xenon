from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["RandomTreesEmbedding"]

class RandomTreesEmbedding(XenonFeatureEngineerAlgorithm):
    module__ = "sklearn.ensemble"
    class__ = "RandomTreesEmbedding"
