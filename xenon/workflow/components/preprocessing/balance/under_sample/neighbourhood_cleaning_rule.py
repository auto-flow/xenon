from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["NeighbourhoodCleaningRule"]


class NeighbourhoodCleaningRule(XenonDataProcessAlgorithm):
    class__ = "NeighbourhoodCleaningRule"
    module__ = "imblearn.under_sampling"
