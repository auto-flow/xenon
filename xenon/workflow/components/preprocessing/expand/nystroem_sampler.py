from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["Nystroem"]

class Nystroem(XenonFeatureEngineerAlgorithm):
    class__ = "Nystroem"
    module__ = "sklearn.kernel_approximation"