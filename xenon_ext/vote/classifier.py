import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class VoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X=None, y=None):
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        probas = [model.predict_proba(X) for model in self.models]
        probas_arr = np.array(probas)
        proba = np.average(probas_arr, axis=0)
        return proba
