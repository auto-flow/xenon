from xenon.pipeline.components.regression_base import XenonRegressionAlgorithm


class ExtraTreesRegressor(
    XenonRegressionAlgorithm,
):
    module__ = "sklearn.ensemble"
    class__ = "ETR"
