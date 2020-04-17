from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["CondensedNearestNeighbour"]


class CondensedNearestNeighbour(XenonDataProcessAlgorithm):
    class__ = "CondensedNearestNeighbour"
    module__ = "imblearn.under_sampling"
