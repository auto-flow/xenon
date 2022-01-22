from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["SMOTENC"]


class SMOTENC(XenonDataProcessAlgorithm):
    class__ = "SMOTENC"
    module__ = "imblearn.over_sampling"
