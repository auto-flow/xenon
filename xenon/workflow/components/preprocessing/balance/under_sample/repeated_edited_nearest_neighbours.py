from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["RepeatedEditedNearestNeighbours"]


class RepeatedEditedNearestNeighbours(XenonDataProcessAlgorithm):
    class__ = "RepeatedEditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
