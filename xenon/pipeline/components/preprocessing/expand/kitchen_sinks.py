from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["RandomKitchenSinks"]

class RandomKitchenSinks(XenonFeatureEngineerAlgorithm):
    module__ = "sklearn.kernel_approximation"
    class__ = "RBFSampler"