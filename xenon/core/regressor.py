from sklearn.base import RegressorMixin

from xenon.core.base import XenonEstimator

__all__ = ["XenonRegressor"]


class XenonRegressor(XenonEstimator, RegressorMixin):
    checked_mainTask = "regression"

    def predict(
            self,
            X_test
    ):
        self._predict(X_test)
        return self.estimator.predict(self.data_manager.X_test)
