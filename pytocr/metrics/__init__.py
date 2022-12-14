import copy

__all__ = ["build_metric"]

from .det_metric import DetMetric
from .rec_metric import RecMetric
from .cls_metric import ClsMetric
from .distillation_metric import DistillationMetric
from .table_metric import TableMetric


def build_metric(config):
    support_dict = [
        "DetMetric", "RecMetric", "ClsMetric", "DistillationMetric", 
        "TableMetric"
        ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
