from xenon.data_container import DataFrameContainer
from xenon.utils.data import softmax
from xenon.workflow.components.base import XenonIterComponent
from xenon.workflow.components.classification_base import XenonClassificationAlgorithm

__all__ = ["SGDClassifier"]


class SGDClassifier(XenonIterComponent, XenonClassificationAlgorithm):
    module__ = "sklearn.linear_model.stochastic_gradient"
    class__ = "SGDClassifier"
    iterations_name = "max_iter"

    def predict_proba(self, X: DataFrameContainer):
        if self.hyperparams["loss"] in ["log", "modified_huber"]:
            return super(SGDClassifier, self).predict_proba(X)
        else:
            df = self.component.decision_function(X.data)
            return softmax(df)
