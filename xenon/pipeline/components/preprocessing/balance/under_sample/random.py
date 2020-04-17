from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["RandomUnderSampler"]


class RandomUnderSampler(XenonDataProcessAlgorithm):
    class__ = "RandomUnderSampler"
    module__ = "imblearn.under_sampling"
