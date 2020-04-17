from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm

__all__ = ["AdaboostRegressor"]

class AdaboostRegressor(XenonRegressionAlgorithm):
    class__ = "AdaBoostRegressor"
    module__ = "sklearn.ensemble"

    def after_process_hyperparams(self, hyperparams):
        import sklearn.tree
        hyperparams = super(AdaboostRegressor, self).after_process_hyperparams(hyperparams)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=hyperparams.pop("max_depth"))
        hyperparams.update({
            "base_estimator": base_estimator
        })
        return hyperparams
