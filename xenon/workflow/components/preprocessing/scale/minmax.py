from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["MinMaxScaler"]

class MinMaxScaler(XenonFeatureEngineerAlgorithm):
    class__ = "MinMaxScaler"
    module__ = "sklearn.preprocessing"


