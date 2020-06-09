from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["KMeansSMOTE"]


class KMeansSMOTE(XenonDataProcessAlgorithm):
    class__ = "KMeansSMOTE"
    module__ = "imblearn.over_sampling"
