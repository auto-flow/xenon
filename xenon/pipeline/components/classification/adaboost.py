from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm

__all__ = ["AdaboostClassifier"]


class AdaboostClassifier(XenonClassificationAlgorithm):
    class__ = "AdaBoostClassifier"
    module__ = "sklearn.ensemble"
    tree_model = True

    def after_process_hyperparams(self, hyperparams):
        import sklearn.tree
        hyperparams = super(AdaboostClassifier, self).after_process_hyperparams(hyperparams)
        base_estimator = sklearn.tree.DecisionTreeClassifier(max_depth=hyperparams.pop("max_depth"))
        hyperparams.update({
            "base_estimator": base_estimator
        })
        return hyperparams
