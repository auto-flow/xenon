from sklearn.base import ClassifierMixin

from xenon.core.base import XenonEstimator

__all__ = ["XenonClassifier"]


class XenonClassifier(XenonEstimator, ClassifierMixin):
    checked_mainTask = "classification"

    def predict(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict(self.data_manager.X_test)

    def predict_proba(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict_proba(self.data_manager.X_test)

    def score(self, X, y, sample_weight=None):
        y=self.data_manager.encode_label(y)
        return super(XenonClassifier, self).score(X, y)
