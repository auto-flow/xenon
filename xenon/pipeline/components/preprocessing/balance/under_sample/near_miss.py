from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["NearMiss"]


class NearMiss(XenonDataProcessAlgorithm):
    class__ = "NearMiss"
    module__ = "imblearn.under_sampling"
