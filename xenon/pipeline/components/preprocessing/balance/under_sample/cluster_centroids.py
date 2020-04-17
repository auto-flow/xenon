from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["ClusterCentroids"]


class ClusterCentroids(XenonDataProcessAlgorithm):
    class__ = "ClusterCentroids"
    module__ = "imblearn.under_sampling"
