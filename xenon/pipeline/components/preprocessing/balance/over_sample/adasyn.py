from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["ADASYN"]


class ADASYN(XenonDataProcessAlgorithm):
    class__ = "ADASYN"
    module__ = "imblearn.over_sampling"
