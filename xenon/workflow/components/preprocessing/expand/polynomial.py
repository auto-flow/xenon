from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__ = ["PolynomialFeatures"]


class PolynomialFeatures(XenonFeatureEngineerAlgorithm):
    module__ = "sklearn.preprocessing"
    class__ = "PolynomialFeatures"
