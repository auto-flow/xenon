from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["OneSidedSelection"]


class OneSidedSelection(XenonDataProcessAlgorithm):
    class__ = "OneSidedSelection"
    module__ = "imblearn.under_sampling"
