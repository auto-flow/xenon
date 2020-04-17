from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["RandomOverSampler"]


class RandomOverSampler(XenonDataProcessAlgorithm):
    class__ = "RandomOverSampler"
    module__ = "imblearn.over_sampling"
