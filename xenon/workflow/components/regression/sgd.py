from copy import deepcopy

from sklearn.preprocessing import StandardScaler

from xenon.data_container import NdArrayContainer
from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["SGDRegressor"]


class SGDRegressor(XenonIterComponent, XenonRegressionAlgorithm):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDRegressor"
    iterations_name = "max_iter"

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
