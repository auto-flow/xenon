from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["SVMSMOTE"]


class SVMSMOTE(XenonDataProcessAlgorithm):
    class__ = "SVMSMOTE"
    module__ = "imblearn.over_sampling"
