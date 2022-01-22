from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["TomekLinks"]


class TomekLinks(XenonDataProcessAlgorithm):
    class__ = "TomekLinks"
    module__ = "imblearn.under_sampling"
