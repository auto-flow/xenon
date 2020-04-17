from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["EditedNearestNeighbours"]


class EditedNearestNeighbours(XenonDataProcessAlgorithm):
    class__ = "EditedNearestNeighbours"
    module__ = "imblearn.under_sampling"
