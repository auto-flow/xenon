from typing import Dict
from importlib import import_module
from xenon.workflow.components.feature_engineer_base import XenonFeatureEngineerAlgorithm

__all__=["FeatureAgglomeration"]

class FeatureAgglomeration(XenonFeatureEngineerAlgorithm):
    module__ = "sklearn.cluster"
    class__ = "FeatureAgglomeration"

    def after_process_hyperparams(self, hyperparams) -> Dict:
        hyperparams=super(FeatureAgglomeration, self).after_process_hyperparams(hyperparams)
        pooling_func=hyperparams["pooling_func"]
        module,func=pooling_func.split(".")
        M=import_module(module)
        func=getattr(M,func)
        hyperparams["pooling_func"]=func
        return hyperparams
