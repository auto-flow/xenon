from xenon.workflow.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["InstanceHardnessThreshold"]


class InstanceHardnessThreshold(XenonDataProcessAlgorithm):
    class__ = "InstanceHardnessThreshold"
    module__ = "imblearn.under_sampling"
