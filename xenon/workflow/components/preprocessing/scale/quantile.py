from typing import Dict

from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["QuantileTransformer"]

class QuantileTransformer(XenonFeatureEngineerAlgorithm):
    class__ = "QuantileTransformer"
    module__ = "sklearn.preprocessing"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(QuantileTransformer, self).after_process_hyperparams(hyperparams)
        hyperparams["n_quantiles"]=min(self.shape[1],hyperparams["n_quantiles"])
        return hyperparams

