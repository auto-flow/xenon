import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator


class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        pred_arr = np.array(preds)
        pred = np.average(pred_arr, axis=0)
        return pred
