from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["LGBMClassifier"]


class LGBMClassifier(XenonClassificationAlgorithm):
    class__ = "LGBMClassifier"
    module__ = "xenon.estimator.wrap_lightgbm"

    boost_model = True
    tree_model = True
    support_early_stopping = True

    def core_fit(self, estimator, X, y=None, X_valid=None, y_valid=None, X_test=None,
                 y_test=None, feature_groups=None):
        use_categorical_feature = self.hyperparams.get("use_categorical_feature", False)
        categorical_features_indices = "auto"  # todo: 配合 OrdinalEncoder
        return self.component.fit(
            X, y, X_valid, y_valid, categorical_feature=categorical_features_indices,
        )
