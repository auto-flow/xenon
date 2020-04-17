import numpy as np

from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm
from xenon.utils.data import softmax

__all__=["SGD"]


class SGD(
    XenonClassificationAlgorithm
):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"

    def predict_proba(self, X):
        if self.hyperparams["loss"] in ["log", "modified_huber"]:
            return super(SGD, self).predict_proba(X)
        else:
            df = self.estimator.decision_function(X)
            return softmax(df)

