from xenon.lazy_import import Configuration

from xenon.utils.klass import StrSignatureMixin


class BaseEvaluator(StrSignatureMixin):
    def init_data(self,**kwargs):
        pass

    def __call__(self, shp:Configuration):
        pass