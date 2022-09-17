import importlib

from .det_metric import DetMetric
from .rec_metric import RecMetric
from .cls_metric import ClsMetric


class DistillationMetric(object):
    def __init__(self,
                 keys=None,
                 base_metric_name=None,
                 main_indicator=None,
                 **kwargs):
        self.main_indicator = main_indicator
        if not isinstance(keys, list):
            self.keys = [keys]
        else:
            self.keys = keys
        self.base_metric_name = base_metric_name
        self.kwargs = kwargs
        self.metrics = None

    def _init_metrcis(self, preds):
        self.metrics = dict()
        mod = importlib.import_module(__name__)
        for key in preds:
            self.metrics[key] = getattr(mod, self.base_metric_name)(
                main_indicator=self.main_indicator, **self.kwargs)
            self.metrics[key].reset()

    def __call__(self, preds, batch, **kwargs):
        assert isinstance(preds, dict)
        if self.metrics is None:
            self._init_metrcis(preds)
        for key in preds:
            self.metrics[key].__call__(preds[key], batch, **kwargs)

    def get_metric(self):
        """
        """
        output = dict()
        best_main_indicator = -1
        for key in self.metrics:
            metric = self.metrics[key].get_metric()
            # main indicator
            # 两个学生模型同等看待，哪个学的好用哪个
            if key in self.keys:
                if metric[self.main_indicator] > best_main_indicator:
                    best_main_indicator = metric[self.main_indicator]
                    output.update(metric)
            for sub_key in metric:
                output["{}_{}".format(key, sub_key)] = metric[sub_key]
        return output

    def reset(self):
        for key in self.metrics:
            self.metrics[key].reset()