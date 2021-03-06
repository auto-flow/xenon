from copy import deepcopy

from xenon.workflow.components.regression_base import XenonRegressionAlgorithm

__all__ = ["GaussianProcess"]


class GaussianProcess(XenonRegressionAlgorithm):
    class__ = "GaussianProcessRegressor"
    module__ = "sklearn.gaussian_process"

    def after_process_hyperparams(self, hyperparams):
        from sklearn.gaussian_process.kernels import RBF
        hyperparams = deepcopy(hyperparams)
        thetaL = hyperparams.pop("thetaL")
        thetaU = hyperparams.pop("thetaU")
        n_features = self.shape[1]
        kernel = RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=[(thetaL, thetaU)] * n_features
        )
        hyperparams.update({
            "kernel": kernel
        })
        return hyperparams
