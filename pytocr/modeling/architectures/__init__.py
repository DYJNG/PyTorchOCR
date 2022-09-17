import copy

from .base_model import BaseModel
from .distillation_model import DistillationModel

__all__ = ["build_model"]


def build_model(config):
    config = copy.deepcopy(config)
    if not "name" in config:
        arch = BaseModel(config)
    else:
        support_dict = ["DistillationModel"]
        name = config.pop("name")
        assert name in support_dict, Exception(
            "architecture only support {}".format(support_dict))
        arch = eval(name)(config)
    return arch