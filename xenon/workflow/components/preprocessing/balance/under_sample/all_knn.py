from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["AllKNN"]


class AllKNN(XenonDataProcessAlgorithm):
    class__ = "AllKNN"
    module__ = "imblearn.under_sampling"
