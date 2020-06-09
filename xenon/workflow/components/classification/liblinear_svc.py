from xenon.data_container import DataFrameContainer
from xenon.utils.data import softmax
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["LibLinear_SVC"]


class LibLinear_SVC(XenonClassificationAlgorithm):
    class__ = "LinearSVC"
    module__ = "sklearn.svm"
    OVR__ = True

    def predict_proba(self, X: DataFrameContainer):
        df = self.component.decision_function(X.data)
        return softmax(df)