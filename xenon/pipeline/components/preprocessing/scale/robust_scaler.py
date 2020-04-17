from typing import Dict

from xenon.pipeline.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["RobustScalerComponent"]

class RobustScalerComponent(XenonFeatureEngineerAlgorithm):
    class__ = "RobustScaler"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(RobustScalerComponent, self).after_process_hyperparams(hyperparams)
        q_min=hyperparams.pop("q_min")
        q_max=hyperparams.pop("q_max")
        hyperparams["quantile_range"]=(q_min,q_max)
        return hyperparams
