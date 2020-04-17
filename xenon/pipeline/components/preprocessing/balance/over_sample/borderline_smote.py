from xenon.pipeline.components.data_process_base import XenonDataProcessAlgorithm

__all__ = ["BorderlineSMOTE"]


class BorderlineSMOTE(XenonDataProcessAlgorithm):
    class__ = "BorderlineSMOTE"
    module__ = "imblearn.over_sampling"
