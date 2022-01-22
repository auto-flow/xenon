from sklearn.base import BaseEstimator


class StackEstimator(BaseEstimator):
    mainTask = None

    def __init__(
            self,
            meta_learner=None,
            use_features_in_secondary=False,
    ):
        self.meta_learner = meta_learner
        self.use_features_in_secondary = use_features_in_secondary
        self.estimators_list = None
        self.prediction_list = None
        assert self.mainTask in ("classification", "regression")

    def predict_meta_features(self, X, is_train):
        raise NotImplementedError

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X, False)
        return predict_fn(meta_features)

    def predict(self, X):
        return self._do_predict(X, self.meta_learner.predict)
