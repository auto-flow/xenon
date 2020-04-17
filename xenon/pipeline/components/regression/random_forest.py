from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class RandomForest(
    XenonRegressionAlgorithm,
):
    class__ = "RandomForestRegressor"
    module__ = "sklearn.ensemble"