from sklearn.calibration import CalibratedClassifierCV

from xenon.pipeline.components.classification_base import XenonClassificationAlgorithm
from xenon.utils.data import softmax

__all__=["LibLinear_SVC"]

class LibLinear_SVC(XenonClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True

    def predict_proba(self, X):
        decision_function=self.estimator.decision_function(X)
        return softmax(decision_function)