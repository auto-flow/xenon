from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["SMOTE"]


class SMOTE(XenonDataProcessAlgorithm):
    class__ = "SMOTE"
    module__ = "imblearn.over_sampling"
