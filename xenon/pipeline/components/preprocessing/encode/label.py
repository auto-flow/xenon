from xenon.pipeline.components.preprocessing.encode.base import BaseEncoder

__all__ = ["LabelEncoder"]


class LabelEncoder(BaseEncoder):
    class__ = "LabelEncoder"
    module__ = "xenon.feature_engineer.encode.label_encode"
