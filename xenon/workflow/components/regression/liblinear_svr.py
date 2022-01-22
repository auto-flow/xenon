from copy import deepcopy

from sklearn.preprocessing import StandardScaler

from xenon.data_container import NdArrayContainer
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["LibLinear_SVR"]


class LibLinear_SVR(XenonRegressionAlgorithm):
    class__ = "LinearSVR"
    module__ = "sklearn.svm"

    def before_fit_y(self, y: NdArrayContainer):
        if y is None:
            return None
        y = deepcopy(y.data)
        self.scaler = StandardScaler(copy=True)
        y = y.ravel().reshape([-1, 1])
        self.scaler.fit(y)
        return self.scaler.transform(y)

    def after_pred_y(self, y):
        return self.scaler.inverse_transform(y)
